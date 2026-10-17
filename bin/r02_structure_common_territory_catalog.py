#!/usr/bin/env python3
"""Freeze the 88-view common comparator catalogue without granting admission.

The catalogue joins the already materialized continuous common representation
to the K02/K07 target identity for every predeclared territory.  It emits no
model fit and never turns an absent per-view scientific admission into a task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


HASH = re.compile(r"[0-9a-f]{64}")
VIEW = re.compile(
    r"chr(?:0[1-9]|1[0-9]|2[0-2])_part(?:01|02)__(?:BLOCK_AND_LD_MARGIN|WHOLE_CHROMOSOME)"
)
SCOPES = {"BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique)


def pinned(path, expected):
    path = Path(path).resolve()
    require(HASH.fullmatch(expected or "") and path.is_file() and sha(path) == expected,
            "Pinned catalogue source differs: " + str(path))
    return path


def write(path, value):
    path = Path(path)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(not path.exists(), "Refuse to overwrite catalogue output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def validate_territories(value):
    require(set(value) == {"schema", "count", "ids", "source"}
            and value["schema"] == "r02_c_graph_territories_v1"
            and value["count"] == 88 and value["ids"] == sorted(value["ids"])
            and len(value["ids"]) == len(set(value["ids"])) == 88
            and all(VIEW.fullmatch(item) for item in value["ids"]),
            "Exactly the frozen 88 territorial views are required")
    pairs = {}
    for view_id in value["ids"]:
        block, scope = view_id.rsplit("__", 1)
        pairs.setdefault(block, set()).add(scope)
    require(len(pairs) == 44 and all(scopes == SCOPES for scopes in pairs.values()),
            "Each of 44 blocks needs both frozen exclusion scopes")
    return value["ids"]


def target_products(inventory, views):
    require(inventory.get("schema") == "r02_unique_useful_products_archive_v1"
            and inventory.get("source_deleted") is False
            and inventory.get("nextflow_cache_automatically_restored") is False
            and HASH.fullmatch(inventory.get("archive_sha256", "")),
            "Authenticated common unique-products inventory required")
    by_path = {row.get("path"): row for row in inventory.get("entries", [])}
    require(len(by_path) == len(inventory.get("entries", [])), "Duplicate inventory path")
    result = {}
    for view_id in views:
        prefix = "published/targets/target_" + view_id + "/"
        products = {}
        for name in ("manifest.json", "continuous.private.npz"):
            row = by_path.get(prefix + name)
            require(row is not None and row.get("disposition") == "PRESERVE"
                    and row.get("archive_member") and HASH.fullmatch(row.get("sha256", ""))
                    and type(row.get("bytes")) is int and row["bytes"] > 0,
                    "Missing authenticated continuous product for " + view_id + ": " + name)
            products[name] = {key: row[key] for key in
                              ("archive_member", "bytes", "sha256")}
        result[view_id] = products
    return result


def validate_policy(request):
    policy = request.get("policy", {})
    require(request.get("schema") == "r02_k03_common_finite_search_request_v1"
            and request.get("evidence_scope") == "DNABR_R02"
            and set(policy) == {"epsilon", "subdivisions_per_decade", "guard_decades",
                                "max_penalties", "plausibility_rule", "selection_rule",
                                "boundary_action"}
            and policy["plausibility_rule"] == "LOWER_BOUND_LE_MINIMUM_UPPER_BOUND"
            and policy["selection_rule"] == "RETAIN_NUMERICALLY_PLAUSIBLE_SET"
            and policy["boundary_action"] == "REQUIRE_EXTENSION_NO_SELECTION",
            "Exact frozen finite-search policy source required")
    require(request.get("parameters") and request.get("resources")
            and HASH.fullmatch(request.get("scientific_spec_sha256", "")),
            "Frozen search parameters/resources/specification required")
    return {"policy": policy, "parameters": request["parameters"],
            "resources": request["resources"],
            "scientific_spec_sha256": request["scientific_spec_sha256"]}


def validate_assembly_sources(value, views):
    require(value.get("schema") == "r02_common_territory_assembly_sources_v1"
            and value.get("status") == "GCS_SOURCES_AUTHENTICATED_NOT_STAGED_NOT_ADMITTED"
            and value.get("view_count") == 88,
            "Authenticated 88-view K02/K07 assembly source catalogue required")
    rows = value.get("views", [])
    by_view = {row.get("view_id"): row for row in rows}
    require(len(rows) == len(by_view) == 88 and set(by_view) == set(views),
            "Assembly source catalogue must cover the exact frozen 88 views")
    for view_id, row in by_view.items():
        block, scope = view_id.rsplit("__", 1)
        prepared = row.get("prepared", {})
        require(row.get("block_id") == block and row.get("exclusion_scope") == scope
                and prepared.get("directory_uri", "").startswith("gs://")
                and HASH.fullmatch(prepared.get("contract_sha256", ""))
                and HASH.fullmatch(prepared.get("shard_contract_sha256", ""))
                and HASH.fullmatch(prepared.get("shard_manifest_sha256", ""))
                and prepared.get("shard_manifest", {}).get("generation_uri", "").startswith("gs://")
                and prepared.get("shard_objects"),
                "Incomplete authenticated prepared source for " + view_id)
    staging = value.get("staging", {})
    require(type(staging.get("unique_objects")) is int and staging["unique_objects"] > 0
            and type(staging.get("total_bytes")) is int and staging["total_bytes"] > 0,
            "Assembly source staging totals required")
    return by_view


def validate_neural_target_manifest(value, policy_request):
    require(value.get("schema") == "r02_k03_neural_target_adapter_result_v1"
            and value.get("status") == "FIT_SELECT_TARGETS_COMPLETE_NOT_TRAINING_ADMISSION"
            and value.get("source_manifest_sha256") ==
                policy_request.get("source", {}).get("manifest_sha256")
            and set(value.get("targets", {})) == {"FIT", "SELECT"}
            and value.get("evaluate_targets_computed") is False
            and value.get("training_allowed") is False,
            "Existing neural target manifest does not bind the admitted single-view bridge")
    return value


def validate_common_archive_source(value, inventory):
    require(value.get("schema") == "r02_common_archive_source_v1"
            and value.get("status") == "GCS_OBJECT_AUTHENTICATED_READ_ONLY"
            and value.get("uri", "").startswith("gs://")
            and value.get("generation_uri") == value["uri"] + "#" + str(value.get("generation"))
            and str(value.get("generation", "")).isdigit()
            and type(value.get("bytes")) is int and value["bytes"] > 0
            and value.get("sha256") == inventory.get("archive_sha256")
            and value.get("md5_base64") and value.get("crc32c"),
            "Authenticated generation-pinned common archive source required")
    return value


def validate_evaluations(values, prepared_by_view):
    contracts = {scope: {row["prepared"]["contract_sha256"] for row in prepared_by_view.values()
                         if row["exclusion_scope"] == scope} for scope in SCOPES}
    require(all(len(value) == 1 for value in contracts.values()),
            "One exact prepared contract per exclusion scope required")
    scientific = set()
    for scope, evaluation in values.items():
        require(evaluation.get("schema") == "r02_structure_evaluation_config_v1" and
                evaluation.get("prepared_contract_sha256") == next(iter(contracts[scope])) and
                HASH.fullmatch(evaluation.get("scientific_protocol_sha256", "")) and
                type(evaluation.get("min_fit_called")) is int and
                type(evaluation.get("min_pair_markers")) is int,
                "Exact target configuration differs for " + scope)
        scientific.add(evaluation["scientific_protocol_sha256"])
    require(len(scientific) == 1, "Exclusion scopes must retain the same frozen target protocol")
    reference = dict(values["WHOLE_CHROMOSOME"])
    candidate = dict(values["BLOCK_AND_LD_MARGIN"])
    reference.pop("prepared_contract_sha256"); candidate.pop("prepared_contract_sha256")
    require(reference == candidate,
            "LD target rules may differ from WHOLE only in exact prepared contract binding")


def existing_view(bridge_request, bridge_request_sha, bridge_admission,
                  bridge_admission_sha, search_request, search_request_sha,
                  search_admission, search_admission_sha):
    block = bridge_request.get("block_id")
    common = bridge_request.get("sources", {}).get("common", {})
    manifest = common.get("manifest_sha256")
    scope = None
    directory = common.get("directory", "")
    for candidate in SCOPES:
        if directory.endswith("__" + candidate):
            scope = candidate
    view_id = block + "__" + str(scope)
    require(VIEW.fullmatch(view_id) and bridge_request.get("schema") ==
            "r02_k03_common_fit_bridge_request_v1", "Existing bridge view is ambiguous")
    require(bridge_admission == {
        "schema": "r02_k03_common_fit_bridge_admission_v1",
        "status": "COMMON_FIT_INPUT_ADMITTED",
        "evidence_scope": "DNABR_R02",
        "scientific_spec_sha256": bridge_request["scientific_spec_sha256"],
        "request_sha256": bridge_request_sha,
        "evaluate_allowed": False,
    }, "Existing bridge admission does not bind its exact request")
    require(search_request.get("source", {}).get("manifest_sha256")
            and search_admission == {
                "schema": "r02_k03_common_finite_search_admission_v1",
                "status": "FIT_SELECT_FINITE_FAMILY_ADMITTED",
                "evidence_scope": "DNABR_R02",
                "scientific_spec_sha256": search_request["scientific_spec_sha256"],
                "request_sha256": search_request_sha,
                "evaluate_allowed": False,
            }, "Existing finite-search admission does not bind its exact request")
    return view_id, manifest, {
        "scope": "PUBLISHED_SINGLE_VIEW_ONLY",
        "bridge_request_sha256": bridge_request_sha,
        "bridge_admission_sha256": bridge_admission_sha,
        "search_request_sha256": search_request_sha,
        "search_admission_sha256": search_admission_sha,
        "evaluate_allowed": False,
        "reported_completed_adjustments": 23,
        "transfer_to_other_views_allowed": False,
    }


def build(*, territories_path, territories_sha, inventory_path, inventory_sha,
          policy_path, policy_sha, bridge_request_path, bridge_request_sha,
          bridge_admission_path, bridge_admission_sha, search_admission_path,
          search_admission_sha, assembly_sources_path, assembly_sources_sha,
          common_archive_source_path, common_archive_source_sha,
          whole_evaluation_path, whole_evaluation_sha,
          ld_evaluation_path, ld_evaluation_sha,
          neural_target_manifest_path, neural_target_manifest_sha, output):
    sources = {}
    for name, path, digest in (
        ("territories", territories_path, territories_sha),
        ("inventory", inventory_path, inventory_sha),
        ("policy_request", policy_path, policy_sha),
        ("bridge_request", bridge_request_path, bridge_request_sha),
        ("bridge_admission", bridge_admission_path, bridge_admission_sha),
        ("search_admission", search_admission_path, search_admission_sha),
        ("assembly_sources", assembly_sources_path, assembly_sources_sha),
        ("common_archive_source", common_archive_source_path, common_archive_source_sha),
        ("whole_evaluation", whole_evaluation_path, whole_evaluation_sha),
        ("ld_evaluation", ld_evaluation_path, ld_evaluation_sha),
        ("neural_target_manifest", neural_target_manifest_path, neural_target_manifest_sha),
    ):
        source = pinned(path, digest)
        sources[name] = {"path": str(source), "sha256": digest}
    territories = load(sources["territories"]["path"])
    inventory = load(sources["inventory"]["path"])
    policy_request = load(sources["policy_request"]["path"])
    bridge_request = load(sources["bridge_request"]["path"])
    bridge_admission = load(sources["bridge_admission"]["path"])
    search_admission = load(sources["search_admission"]["path"])
    assembly_sources = load(sources["assembly_sources"]["path"])
    common_archive_source = load(sources["common_archive_source"]["path"])
    neural_target_manifest = load(sources["neural_target_manifest"]["path"])
    evaluations = {"WHOLE_CHROMOSOME": load(sources["whole_evaluation"]["path"]),
                   "BLOCK_AND_LD_MARGIN": load(sources["ld_evaluation"]["path"])}
    views = validate_territories(territories)
    products = target_products(inventory, views)
    validate_common_archive_source(common_archive_source, inventory)
    policy = validate_policy(policy_request)
    prepared_by_view = validate_assembly_sources(assembly_sources, views)
    validate_evaluations(evaluations, prepared_by_view)
    validate_neural_target_manifest(neural_target_manifest, policy_request)
    prior_view, prior_manifest, prior = existing_view(
        bridge_request, bridge_request_sha, bridge_admission, bridge_admission_sha,
        policy_request, policy_sha, search_admission, search_admission_sha)
    require(prior_view in products and products[prior_view]["manifest.json"]["sha256"] == prior_manifest,
            "Published single-view bridge is not the catalogued common representation")

    out = Path(output).resolve()
    require(not out.exists() and out.parent.is_dir(), "New catalogue output directory required")
    out.mkdir(mode=0o700)
    requests = out / "binding-requests"
    requests.mkdir()
    catalogue_views = []
    request_rows = []
    for view_id in views:
        block, scope = view_id.rsplit("__", 1)
        admitted = view_id == prior_view
        prepared = prepared_by_view[view_id]["prepared"]
        evaluation_source = sources["whole_evaluation" if scope == "WHOLE_CHROMOSOME"
                                    else "ld_evaluation"]
        request = {
            "schema": "r02_common_territory_binding_request_v1",
            "status": ("EXISTING_SINGLE_VIEW_FIT_SELECT_ADMISSION_RECORDED" if admitted else
                       "SOURCES_AUTHENTICATED_PER_VIEW_ADMISSIONS_REQUIRED"),
            "view_id": view_id,
            "block_id": block,
            "exclusion_scope": scope,
            "common_representation": {
                "kind": "FULL_NUMERICAL_COMMON_SPECTRUM_NOT_PREDICTIVE_SELECTION",
                "archive_source": common_archive_source,
                "products": products[view_id],
            },
            "target_source": {
                "identity": "K02_K07_PREPARED_PAIR_TARGET_SAME_BLOCK_AND_EXCLUSION",
                "state": "GCS_SOURCE_AUTHENTICATED_NOT_STAGED_NOT_ADMITTED",
                "directory_uri": prepared["directory_uri"],
                "prepared_contract_sha256": prepared["contract_sha256"],
                "shard_path": prepared["shard_path"],
                "shard_contract_sha256": prepared["shard_contract_sha256"],
                "shard_manifest_sha256": prepared["shard_manifest_sha256"],
                "shard_manifest_generation_uri": prepared["shard_manifest"]["generation_uri"],
                "required_roles": ["FIT", "SELECT"],
                "evaluation_configuration": {
                    "state": "EXACT_PREPARED_CONTRACT_BINDING_IDENTIFIED",
                    "path": evaluation_source["path"],
                    "sha256": evaluation_source["sha256"],
                    "applies_to_prepared_contract_sha256": prepared["contract_sha256"],
                },
                "evaluate_target_read_allowed": False,
            },
            "roles": {
                "FIT": {"people": 1460, "use": "derive_gram_and_fit_complete_finite_family"},
                "SELECT": {"people": 476, "use": "apply_frozen_plausibility_and_boundary_rules"},
                "EVALUATE": {"people": 683, "use": "CLOSED_NOT_READ_OR_SCORED"},
            },
            "policy_source_sha256": policy_sha,
            "neural_target_consumer": {
                "workflow": "workflows/r02_k03_neural_target_adapter.nf",
                "source_schema": "r02_k03_fit_select_arrays_v1",
                "source_stage": "FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS",
                "emitted_roles": ["FIT", "SELECT"],
                "training_allowed": False,
                "evaluate_allowed": False,
            },
            "penalty_free_target_producer": {
                "code": "bin/r02_structure_common_target_bridge.py",
                "workflow": "workflows/r02_structure_common_target_bridge.nf",
                "request_materializer": "bin/r02_structure_common_target_requests.py",
                "penalty_present": False,
                "model_fit_performed": False,
            },
            "admission": prior if admitted else {
                "status": "NOT_GRANTED",
                "required": ["COMMON_FIT_INPUT_ADMISSION_FOR_THIS_VIEW",
                             "FIT_SELECT_FINITE_FAMILY_ADMISSION_FOR_THIS_VIEW"],
                "evaluate_allowed": False,
            },
            "runnable": False,
        }
        request_path = requests / (view_id + ".json")
        request_hash = write(request_path, request)
        request_rows.append({"id": view_id, "request": str(request_path),
                             "request_sha256": request_hash, "runnable": False})
        catalogue_views.append({
            "view_id": view_id, "block_id": block, "exclusion_scope": scope,
            "common_representation_manifest_sha256": products[view_id]["manifest.json"]["sha256"],
            "common_archive_generation_uri": common_archive_source["generation_uri"],
            "target_identity": "K02_K07_PREPARED_PAIR_TARGET_SAME_BLOCK_AND_EXCLUSION",
            "prepared_contract_sha256": prepared["contract_sha256"],
            "prepared_shard_manifest_sha256": prepared["shard_manifest_sha256"],
            "target_source_state": "GCS_SOURCE_AUTHENTICATED_NOT_STAGED_NOT_ADMITTED",
            "evaluation_configuration_state": "EXACT_CONTRACT_BINDING_IDENTIFIED",
            "roles_identity": "DNABR_R02_1460_FIT_476_SELECT_683_EVALUATE",
            "finite_search_policy_sha256": policy_sha,
            "admission_state": ("EXISTING_SINGLE_VIEW_ONLY" if admitted else "NOT_GRANTED"),
            "evaluate_state": "CLOSED",
            "request_sha256": request_hash,
        })

    policy_record = {
        "schema": "r02_common_territory_frozen_policy_reference_v1",
        "status": "EXACT_POLICY_IDENTIFIED_REUSE_REQUIRES_PER_VIEW_ADMISSION",
        "source_request_sha256": policy_sha,
        **policy,
        "apply_to_unadmitted_views": False,
        "evaluate_allowed": False,
    }
    policy_record_sha = write(out / "policy.json", policy_record)
    catalogue = {
        "schema": "r02_common_territory_catalogue_v1",
        "status": "CATALOGUED_88_AUTHENTICATED_SOURCE_BINDINGS_ADMISSIONS_INCOMPLETE",
        "evidence_scope": "DNABR_R02",
        "view_count": 88,
        "block_count": 44,
        "scope_counts": {scope: sum(v.endswith("__" + scope) for v in views) for scope in sorted(SCOPES)},
        "role_counts": {"FIT": 1460, "SELECT": 476, "EVALUATE": 683},
        "evaluate_allowed": False,
        "views": catalogue_views,
    }
    catalogue_sha = write(out / "catalogue.json", catalogue)
    alignment = {
        "schema": "r02_k02_k07_common_territory_alignment_v1",
        "status": "ONE_SHARED_CATALOGUE_NO_DIVERGENT_TERRITORY_LIST",
        "catalogue_sha256": catalogue_sha,
        "view_count": 88,
        "consumers": {
            "K02": "continuous_structure_common_representation_and_same_pair_target",
            "K07_B0": "same_common_representation_target_roles_and_exclusion",
            "K07_B1_B2": "same_target_roles_and_exclusion_with_separately_admitted_rare_predictors",
            "K03_COMMON": "same_continuous_common_representation_and_pair_target",
            "STRUCTURE_NEURAL_TARGET": "FIT_SELECT_targets_from_each_admitted_K03_bridge_only",
        },
        "views": [{"view_id": row["view_id"],
                   "common_representation_manifest_sha256": row["common_representation_manifest_sha256"],
                   "target_identity": row["target_identity"]} for row in catalogue_views],
        "evaluate_allowed": False,
    }
    alignment_sha = write(out / "k02-k07-alignment.json", alignment)
    neural_handoff = {
        "schema": "r02_common_territory_neural_target_handoff_v1",
        "status": "ONE_EXISTING_TARGET_READY_87_WAIT_FOR_PER_VIEW_BRIDGES_AND_ADMISSIONS",
        "catalogue_sha256": catalogue_sha,
        "consumer_workflow": "workflows/r02_k03_neural_target_adapter.nf",
        "consumer_code": "bin/r02_k03_neural_target_adapter.py",
        "required_source_schema": "r02_k03_fit_select_arrays_v1",
        "required_source_stage": "FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS",
        "output_roles": ["FIT", "SELECT"],
        "ready": [{"view_id": prior_view,
                   "source_manifest_sha256": neural_target_manifest["source_manifest_sha256"],
                   "neural_target_manifest_sha256": neural_target_manifest_sha}],
        "blocked": [{"view_id": row["view_id"],
                     "reason": ("EVALUATION_CONFIGURATION_AND_PER_VIEW_K03_BRIDGE_ADMISSIONS_REQUIRED"
                                if row["evaluation_configuration_state"] ==
                                "MISSING_FOR_PREPARED_CONTRACT" else
                                "PER_VIEW_K03_BRIDGE_AND_ADMISSIONS_NOT_GRANTED")}
                    for row in catalogue_views if row["view_id"] != prior_view],
        "training_allowed": False,
        "evaluate_allowed": False,
    }
    neural_handoff_sha = write(out / "neural-target-handoff.json", neural_handoff)
    batch = {
        "schema": "r02_common_territory_binding_batch_v1",
        "status": "NO_NEW_RUNNABLE_TASKS_UNTIL_EXPLICIT_ADMISSIONS",
        "catalogue_sha256": catalogue_sha,
        "tasks": [],
        "blocked_requests": request_rows,
        "evaluate_allowed": False,
    }
    batch_sha = write(out / "batch.json", batch)
    missing = [row["view_id"] for row in catalogue_views if row["admission_state"] == "NOT_GRANTED"]
    diagnostic = {
        "schema": "r02_common_territory_catalogue_diagnostic_v1",
        "status": "SOURCES_AND_TARGET_RULES_COMPLETE_PER_VIEW_ADMISSIONS_INCOMPLETE",
        "catalogued_views": 88,
        "continuous_representations_authenticated": 88,
        "prepared_target_sources_authenticated": 88,
        "views_with_exact_evaluation_configuration_binding": sum(
            row["evaluation_configuration_state"] == "EXACT_CONTRACT_BINDING_IDENTIFIED"
            for row in catalogue_views),
        "views_missing_evaluation_configuration_binding": [
            row["view_id"] for row in catalogue_views
            if row["evaluation_configuration_state"] == "MISSING_FOR_PREPARED_CONTRACT"],
        "existing_single_view_admission_recorded": prior_view,
        "existing_single_view_completed_adjustments": 23,
        "new_runnable_views": 0,
        "missing_view_count": len(missing),
        "missing_views": missing,
        "missing_requirements": [
            "stage the authenticated common representations and prepared targets on the execution filesystem",
            "materialize exact per-view penalty-free target bridge requests after staging",
            "common FIT/SELECT target input admission bound to each exact penalty-free bridge request",
            "finite FIT/SELECT family admission bound to each exact search request",
            "separate later admission before any EVALUATE target is read or scored",
        ],
        "policy_not_invented": True,
        "evaluate_allowed": False,
    }
    diagnostic_sha = write(out / "diagnostic.json", diagnostic)
    outputs = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
    manifest = {
        "schema": "r02_common_territory_catalogue_receipt_v1",
        "stage": "INDEPENDENT_CATALOGUE_COMPLETE_NEW_EXECUTION_BLOCKED",
        "catalogue_sha256": catalogue_sha,
        "policy_sha256": policy_record_sha,
        "alignment_sha256": alignment_sha,
        "neural_target_handoff_sha256": neural_handoff_sha,
        "batch_sha256": batch_sha,
        "diagnostic_sha256": diagnostic_sha,
        "source_pins": sources,
        "outputs_sha256": outputs,
        "model_fits_run": False,
        "evaluate_targets_computed": False,
        "biological_validation": False,
    }
    write(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("territories", "territories-sha256", "inventory", "inventory-sha256",
                 "policy-request", "policy-request-sha256", "bridge-request",
                 "bridge-request-sha256", "bridge-admission", "bridge-admission-sha256",
                 "search-admission", "search-admission-sha256", "assembly-sources",
                 "assembly-sources-sha256", "neural-target-manifest",
                 "neural-target-manifest-sha256", "common-archive-source",
                 "common-archive-source-sha256", "whole-evaluation",
                 "whole-evaluation-sha256", "ld-evaluation", "ld-evaluation-sha256",
                 "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = build(
        territories_path=args.territories, territories_sha=args.territories_sha256,
        inventory_path=args.inventory, inventory_sha=args.inventory_sha256,
        policy_path=args.policy_request, policy_sha=args.policy_request_sha256,
        bridge_request_path=args.bridge_request, bridge_request_sha=args.bridge_request_sha256,
        bridge_admission_path=args.bridge_admission, bridge_admission_sha=args.bridge_admission_sha256,
        search_admission_path=args.search_admission, search_admission_sha=args.search_admission_sha256,
        assembly_sources_path=args.assembly_sources,
        assembly_sources_sha=args.assembly_sources_sha256,
        common_archive_source_path=args.common_archive_source,
        common_archive_source_sha=args.common_archive_source_sha256,
        whole_evaluation_path=args.whole_evaluation,
        whole_evaluation_sha=args.whole_evaluation_sha256,
        ld_evaluation_path=args.ld_evaluation,
        ld_evaluation_sha=args.ld_evaluation_sha256,
        neural_target_manifest_path=args.neural_target_manifest,
        neural_target_manifest_sha=args.neural_target_manifest_sha256,
        output=args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
