#!/usr/bin/env python3
"""Plan 960 development link wrappers and ten FIT-only LAI contracts.

Planning requires the final DEVELOPMENT protocol and the verified completion
receipt for the protocol chromosome. Materialization is a metadata-only operation delegated to
``r02_lai_training_campaign.create_training_link`` and additionally requires an
exact admission record.  Neither command loads arrays or opens SELECT/EVALUATE.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import tempfile

import r02_lai_training_campaign as campaign


SCHEMA = "r02_lai_fit_training_link_request_plan_v3"
ADMISSION_SCHEMA = "r02_lai_fit_link_materialization_admission_v2"
RECEIPT_SCHEMA = "r02_lai_fit_link_materialization_receipt_v2"
CONTRACT_PLAN_SCHEMA = "r02_lai_fit_contract_candidate_plan_v2"
CONTRACT_SET_SCHEMA = "r02_lai_fit_contract_set_v2"
CONTRACT_CANDIDATE_SCHEMA = "r02_lai_fit_contract_candidate_v2"
CONTRACT_ADMISSIONS_SCHEMA = "r02_lai_fit_contract_admissions_v2"
FAMILIES = campaign.FAMILIES
CONDITIONS = campaign.CONDITIONS
REGIONS = tuple(f"region-{index:02d}" for index in range(16))


def require(value, message):
    if not value:
        raise ValueError(message)


def spec(path, expected_sha=None):
    path = Path(path).resolve()
    require(path.is_file(), "file absent: " + str(path))
    observed = campaign.sha(path)
    require(expected_sha is None or observed == expected_sha,
            "file SHA-256 differs: " + str(path))
    return {"path": str(path), "sha256": observed}


def write_fresh(path, value):
    path = Path(path)
    require(not path.exists() and path.parent.is_dir(),
            "fresh output file required: " + str(path))
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return value


def normalize_chromosome(value):
    chromosome = str(value)
    require(chromosome.isdigit() and 1 <= int(chromosome) <= 22
            and chromosome == str(int(chromosome)),
            "protocol chromosome must be one canonical autosome from 1 through 22")
    return chromosome


def _completion_value(completion_spec, chromosome):
    chromosome = normalize_chromosome(chromosome)
    value = campaign.read_json(campaign.descriptor(completion_spec, "COMPLETE v2"),
                               completion_spec["sha256"])
    valid_schema = (value.get("schema") ==
                    "r02_lai_factorized_chromosome_prepare_completion_v1" or
                    (chromosome == "22" and value.get("schema") ==
                     "r02_lai_factorized_chr22_prepare_completion_v2"))
    require(valid_schema
            and value.get("status") == "PREPARE_COMPLETE_NO_TRAIN_NO_EVALUATE"
            and value.get("predictors") == 576 and value.get("links") == 2880
            and value.get("combined_condition_batches") == 0
            and value.get("chromosomes") == [chromosome]
            and value.get("training_started") is False
            and value.get("evaluate_opened") is False,
            "completion receipt does not close the exact protocol-chromosome PREPARE inventory")
    return value


def _resolved(protocol, path, expected_sha, expected_family):
    resolved_spec = spec(path, expected_sha)
    value = campaign.read_json(resolved_spec["path"], resolved_spec["sha256"])
    config = campaign.validate_resolved(protocol, value)
    require(config.family == expected_family,
            "resolved configuration is assigned to the wrong LAI family")
    require(float(config.common_radius_cm) == 1.0
            and float(config.rare_radius_cm) == 1.0 and config.width == 64
            and float(config.learning_rate) == 0.0001,
            "integration requires the measured radius-1/width-64/LR-1e-4 recipe")
    return resolved_spec, value, config


def validate_plan_value(value):
    chromosome = normalize_chromosome(value.get("chromosome"))
    from r02_lai_neural_input import parse_case
    require(value.get("schema") == SCHEMA
            and value.get("status") ==
                "READY_FOR_METADATA_WRAPPING_AFTER_EXPLICIT_ADMISSION"
            and value.get("scope") == "DEVELOPMENT_FIT_WITH_RESERVED_SELECT_METADATA"
            and value.get("families") == list(FAMILIES)
            and value.get("conditions") == list(CONDITIONS)
            and value.get("fit_cases_per_family") == 3
            and value.get("regions_per_case") == 16
            and value.get("jobs_total") == 960
            and value.get("select_metadata_bound") is True
            and value.get("select_values_loaded") is False
            and value.get("select_opened") is False
            and value.get("evaluate_opened") is False
            and value.get("scientific_training_started") is False,
            "FIT request plan header differs")
    jobs = value.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 960
            and len({job.get("job_id") for job in jobs}) == 960
            and all(set(job) == {"job_id", "family", "condition", "role", "chromosome",
                                 "case_id", "region_id", "factorized_request",
                                 "source_prepare_receipt", "resolved", "output"}
                    and job["chromosome"] == chromosome
                    and job["family"] in FAMILIES
                    and job["condition"] in CONDITIONS
                    and job["role"] in {"FIT", "SELECT"}
                    and parse_case(job["case_id"])[1] == chromosome
                    and job["region_id"] in REGIONS for job in jobs),
            "FIT request jobs differ or are duplicated")
    by_unit = defaultdict(set)
    outputs = set()
    for job in jobs:
        by_unit[job["family"], job["condition"], job["role"],
                job["case_id"]].add(
            job["region_id"])
        require(job["output"] not in outputs, "two jobs share one wrapper output")
        outputs.add(job["output"])
    for family in FAMILIES:
        for condition in CONDITIONS:
            for role in ("FIT", "SELECT"):
                units = {case: regions for
                         (found_family, found_condition, found_role, case), regions
                         in by_unit.items() if found_family == family
                         and found_condition == condition and found_role == role}
                require(len(units) == 3 and all(regions == set(REGIONS)
                                                 for regions in units.values()),
                        "family-condition-role lacks three cases by 16 regions")
    return value


def build(plan_path, plan_sha256, completion_path, completion_sha256,
          protocol_path, protocol_sha256, prepared_root,
          resolved_cnn, resolved_cnn_sha256, resolved_attention,
          resolved_attention_sha256, wrapper_root, output):
    plan_spec = spec(plan_path, plan_sha256)
    completion_spec = spec(completion_path, completion_sha256)
    protocol_spec = spec(protocol_path, protocol_sha256)
    protocol, decision_sha256 = campaign.validate_protocol(
        protocol_spec["path"], protocol_spec["sha256"])
    chromosome = normalize_chromosome(protocol.get("chromosome"))
    require(protocol["scope"] == "DEVELOPMENT"
            and protocol["input_plan"] == plan_spec
            and protocol["input_completion"] == completion_spec
            and Path(protocol["input_output_root"]).resolve() == Path(prepared_root).resolve(),
            "protocol does not bind these PLAN/COMPLETE-v2/PREPARE paths")
    _completion_value(completion_spec, chromosome)
    plan = campaign.read_json(plan_spec["path"], plan_spec["sha256"])
    require(plan.get("schema") == "r02_lai_factorized_regional_request_plan_v1"
            and plan.get("conditions") == list(CONDITIONS)
            and plan.get("training_authorized") is False
            and plan.get("selection_authorized") is False
            and plan.get("evaluate_opened") is False,
            "factorized plan differs or already opens TRAIN/SELECT/EVALUATE")
    resolved = {
        "lai_cnn": _resolved(protocol, resolved_cnn, resolved_cnn_sha256, "lai_cnn"),
        "lai_local_attention": _resolved(
            protocol, resolved_attention, resolved_attention_sha256,
            "lai_local_attention"),
    }
    links = {(row["predictor_id"], row["condition"]): row
             for row in plan["links"] if str(row.get("chromosome")) == chromosome}
    require(len(links) == 2880,
            "PLAN does not contain 2,880 unique links for the protocol chromosome")
    prepared_root = Path(prepared_root).resolve()
    wrapper_root = Path(wrapper_root).resolve()
    jobs = []
    seen = defaultdict(set)
    for row in plan["predictors"]:
        if str(row.get("chromosome")) != chromosome:
            continue
        request = campaign.read_json(
            campaign.descriptor(row["request"], "predictor request"),
            row["request"]["sha256"])
        family = request.get("family")
        role = request.get("role")
        if role not in {"FIT", "SELECT"} or family not in FAMILIES:
            continue
        resolved_spec, _, config = resolved[family]
        if float(request.get("scale_cm", -1)) != float(config.common_radius_cm):
            continue
        case_id, region_id = request.get("case_id"), request.get("region_id")
        require(request.get("schema") == "r02_lai_regional_predictor_request_v1"
                and request.get("operation") == "PREPARE"
                and request.get("direction") == "FOLD_A_TO_FOLD_B"
                and str(request.get("chromosome")) == chromosome
                and request.get("training_allowed") is False
                and request.get("selection_performed") is False
                and request.get("evaluate_opened") is False
                and region_id in REGIONS,
                "FIT predictor request identity differs")
        from r02_lai_neural_input import parse_case
        scenario, case_chromosome, donor_fold, replicate = parse_case(case_id)
        expected_fold = "FOLD_A" if role == "FIT" else "FOLD_B"
        require(scenario == "PULSE_T10_ANCHOR" and case_chromosome ==
                normalize_chromosome(protocol.get("chromosome"))
                and donor_fold == expected_fold and replicate in {1, 2, 3},
                "development request is not one of the six admitted role-realizations")
        seen[family, role, case_id].add(region_id)
        for condition in CONDITIONS:
            link = links.get((row["predictor_id"], condition))
            require(link is not None, "PLAN omits a FIT predictor-condition link")
            directory = prepared_root / "links" / link["link_id"]
            request_spec = spec(directory / "link-request.json")
            receipt_spec = spec(directory / "linked" / "receipt.json")
            source = campaign.read_json(receipt_spec["path"], receipt_spec["sha256"])
            arrays = source.get("arrays", [])
            array_hashes = source.get("reconstructed_arrays_sha256", {})
            require(source.get("schema") ==
                        "r02_lai_regional_truth_factorized_link_receipt_v1"
                    and source.get("status") ==
                        "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
                    and source.get("factorized_link") is True
                    and source.get("truth_joined") is True
                    and source.get("request_sha256") == request_spec["sha256"]
                    and source.get("task_id") == request.get("task_id")
                    and source.get("case_id") == case_id
                    and source.get("role") == role
                    and source.get("condition") == condition
                    and source.get("region_id") == region_id
                    and source.get("training_authorized") is False
                    and source.get("selection_authorized") is False
                    and source.get("evaluate_opened") is False
                    and {"query_valid", "query_core", "truth_observed", "truth_state"}
                        <= set(arrays)
                    and set(arrays) == set(array_hashes)
                    and source.get("person_indices") == list(range(128)),
                    "published PREPARE link identity differs")
            destination = (wrapper_root / family / condition / case_id /
                           (region_id + ".training-link.json"))
            jobs.append({
                "job_id": f"{family}.{case_id}.{region_id}.{condition}",
                "family": family, "condition": condition, "role": role,
                "chromosome": normalize_chromosome(protocol.get("chromosome")),
                "case_id": case_id, "region_id": region_id,
                "factorized_request": request_spec,
                "source_prepare_receipt": receipt_spec,
                "resolved": resolved_spec,
                "output": str(destination),
            })
    require(len(seen) == 12 and all(regions == set(REGIONS) for regions in seen.values()),
            "each family and role requires three cases with all 16 regions")
    jobs.sort(key=lambda job: (FAMILIES.index(job["family"]), job["role"], job["case_id"],
                               job["region_id"], CONDITIONS.index(job["condition"])))
    counts = Counter(job["family"] for job in jobs)
    condition_counts = Counter(job["condition"] for job in jobs)
    role_counts = Counter(job["role"] for job in jobs)
    require(len(jobs) == 960 and counts == {family: 480 for family in FAMILIES}
            and condition_counts == {condition: 192 for condition in CONDITIONS}
            and role_counts == {"FIT": 480, "SELECT": 480},
            "request plan must contain exactly 960 family-condition-role-region jobs")
    value = {
        "schema": SCHEMA,
        "status": "READY_FOR_METADATA_WRAPPING_AFTER_EXPLICIT_ADMISSION",
        "scope": "DEVELOPMENT_FIT_WITH_RESERVED_SELECT_METADATA",
        "chromosome": normalize_chromosome(protocol.get("chromosome")),
        "plan": plan_spec, "completion": completion_spec,
        "protocol": protocol_spec, "protocol_decision_sha256": decision_sha256,
        "prepared_root": str(prepared_root), "wrapper_root": str(wrapper_root),
        "families": list(FAMILIES), "conditions": list(CONDITIONS),
        "fit_cases_per_family": 3, "regions_per_case": 16,
        "jobs_total": 960, "jobs_by_family": dict(counts),
        "jobs_by_condition": dict(condition_counts), "jobs_by_role": dict(role_counts),
        "jobs": jobs, "select_metadata_bound": True,
        "select_values_loaded": False, "select_opened": False, "evaluate_opened": False,
        "nam_final_required_for_fit_wrapping": False,
        "scientific_training_started": False,
    }
    validate_plan_value(value)
    return write_fresh(output, value)


def _validate_admission(admission, plan_spec, chromosome):
    chromosome = normalize_chromosome(chromosome)
    required = {"schema", "status", "scope", "operation", "plan",
                "chromosome", "select_opened", "evaluate_opened"}
    require(set(admission) == required
            and admission["schema"] == ADMISSION_SCHEMA
            and admission["status"] ==
                "ADMIT_DEVELOPMENT_METADATA_WRAPPING_FOR_FIT_ONLY_EXECUTION"
            and admission["scope"] == "DEVELOPMENT_FIT_WITH_RESERVED_SELECT_METADATA"
            and admission["operation"] ==
                "CREATE_PROTOCOL_BOUND_FIT_AND_RESERVED_SELECT_LINK_RECEIPTS"
            and admission["plan"] == plan_spec
            and admission["chromosome"] == chromosome
            and admission["select_opened"] is False
            and admission["evaluate_opened"] is False,
            "admission does not authorize this exact FIT metadata plan")


def materialize(plan_path, plan_sha256, admission_path, admission_sha256, output):
    plan_spec = spec(plan_path, plan_sha256)
    plan = campaign.read_json(plan_spec["path"], plan_spec["sha256"])
    validate_plan_value(plan)
    admission_spec = spec(admission_path, admission_sha256)
    admission = campaign.read_json(admission_spec["path"], admission_spec["sha256"])
    _validate_admission(admission, plan_spec, plan["chromosome"])
    created = reused = 0
    output_specs = []
    for job in plan["jobs"]:
        destination = Path(job["output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            candidate = Path(temporary) / "expected.json"
            campaign.create_training_link(
                job["factorized_request"]["path"], job["factorized_request"]["sha256"],
                job["source_prepare_receipt"]["path"],
                job["source_prepare_receipt"]["sha256"],
                plan["protocol"]["path"], plan["protocol"]["sha256"],
                job["resolved"]["path"], job["resolved"]["sha256"],
                job["role"], job["condition"], candidate)
            expected = candidate.read_bytes()
            if destination.exists():
                require(destination.is_file() and destination.read_bytes() == expected,
                        "existing wrapper differs: " + str(destination))
                reused += 1
            else:
                temporary_output = destination.with_suffix(destination.suffix + ".tmp")
                require(not temporary_output.exists(), "stale temporary wrapper exists")
                temporary_output.write_bytes(expected)
                os.replace(temporary_output, destination)
                created += 1
        output_specs.append(spec(destination))
    value = {
        "schema": RECEIPT_SCHEMA,
        "status": "ALL_960_DEVELOPMENT_TRAINING_LINK_RECEIPTS_COMPLETE",
        "chromosome": plan["chromosome"], "plan": plan_spec, "admission": admission_spec,
        "created": created, "reused_exact": reused, "total": len(output_specs),
        "outputs": output_specs, "arrays_loaded": False,
        "training_started": False, "select_metadata_bound": True,
        "select_values_loaded": False, "select_opened": False, "evaluate_opened": False,
    }
    return write_fresh(output, value)


def _people_from_metadata(jobs):
    people = {}
    seen_predictors = set()
    for job in jobs:
        request = campaign.read_json(job["factorized_request"]["path"],
                                     job["factorized_request"]["sha256"])
        regional_spec = request.get("regional_request")
        regional_path = campaign.descriptor(regional_spec, "regional predictor request")
        if regional_spec["sha256"] in seen_predictors:
            continue
        seen_predictors.add(regional_spec["sha256"])
        regional = campaign.read_json(regional_path, regional_spec["sha256"])
        person_ids = regional.get("expected_person_ids")
        components = regional.get("expected_component_ids")
        require(regional.get("role") == job["role"]
                and isinstance(person_ids, list) and len(person_ids) == 128
                and isinstance(components, list) and len(components) == 128,
                "regional request lacks the 128-person role metadata")
        for person, component in zip(person_ids, components):
            item = {"role": job["role"], "component": component}
            require(person not in people or people[person] == item,
                    "person metadata crosses roles or components")
            people[person] = item
    require({item["role"] for item in people.values()} == {"FIT", "SELECT"},
            "person metadata omits a development role")
    return people


def contract_candidates(plan_path, plan_sha256, materialization_path,
                        materialization_sha256, contract_base_path,
                        contract_base_sha256, output_root, output):
    plan_spec = spec(plan_path, plan_sha256)
    plan = validate_plan_value(campaign.read_json(plan_spec["path"], plan_spec["sha256"]))
    materialization_spec = spec(materialization_path, materialization_sha256)
    materialization = campaign.read_json(materialization_spec["path"],
                                         materialization_spec["sha256"])
    require(materialization.get("schema") == RECEIPT_SCHEMA
            and materialization.get("plan") == plan_spec
            and materialization.get("chromosome") == plan["chromosome"]
            and materialization.get("total") == 960
            and materialization.get("select_metadata_bound") is True
            and materialization.get("select_values_loaded") is False
            and materialization.get("training_started") is False,
            "materialization receipt differs")
    contract_base_spec = spec(contract_base_path, contract_base_sha256)
    base = campaign.read_json(contract_base_spec["path"], contract_base_spec["sha256"])
    generated = {"admission", "persons", "batches", "steps", "threads", "lai_chromosome",
                 "information_condition", "lai_training_protocol", "query_max_gap_cm"}
    require(not (set(base) & generated)
            and base.get("schema") == "r02_context_training_v1"
            and base.get("scope") == "DEVELOPMENT" and base.get("objective") == "lai"
            and base.get("controls_materialized") is True
            and base.get("training_authorized") is False
            and base.get("dataset_declared_scope") == "BOUNDED_CONTEXT_PILOT"
            and base.get("purpose") ==
                "PREPARATION_OR_EXPLICIT_PILOT_NOT_PRIMARY_DELIVERY"
            and set(base.get("limits", {})) ==
                {"max_common_sites", "max_event_sites", "max_targets"},
            "contract base is incomplete, pre-admitted, or contains generated fields")
    protocol = campaign.read_json(plan["protocol"]["path"], plan["protocol"]["sha256"])
    require(protocol["optimization"]["optimizer_updates"] == 96,
            "reviewed integration requires exactly 96 optimizer updates per fit")
    people = _people_from_metadata(plan["jobs"])
    output_root = Path(output_root).resolve()
    contracts = []
    jobs_by_key = defaultdict(list)
    for job in plan["jobs"]:
        jobs_by_key[job["family"], job["condition"], job["role"]].append(job)
    for family in FAMILIES:
        family_jobs = [job for job in plan["jobs"] if job["family"] == family]
        resolved_spec = family_jobs[0]["resolved"]
        resolved = campaign.read_json(resolved_spec["path"], resolved_spec["sha256"])
        require(all(job["resolved"] == resolved_spec for job in family_jobs),
                "one family references multiple resolved configurations")
        for condition in CONDITIONS:
            batches = {}
            for role in ("FIT", "SELECT"):
                selected = sorted(jobs_by_key[family, condition, role],
                                  key=lambda job: (job["case_id"], job["region_id"]))
                require(len(selected) == 48, "contract role requires 48 metadata entries")
                batches[role] = [{
                    "format": "LAI_FACTORIZED_TRAINING_LINK_V1",
                    "protocol_id": "FOLD_A_TO_FOLD_B",
                    "role": role,
                    "lai_case_id": job["case_id"],
                    "path": job["factorized_request"]["path"],
                    "sha256": job["factorized_request"]["sha256"],
                    "receipt_path": job["output"],
                    "receipt_sha256": campaign.sha(job["output"]),
                } for job in selected]
            contract = copy.deepcopy(base)
            contract["persons"] = people
            contract["batches"] = batches
            contract["steps"] = 96
            contract["threads"] = protocol["resources"]["threads"]
            contract["limits"] = {**contract["limits"],
                                  "max_batch_bytes": protocol["resources"]["max_batch_bytes"]}
            contract["query_max_gap_cm"] = resolved.get("query_max_gap_cm")
            contract["lai_chromosome"] = plan["chromosome"]
            contract["information_condition"] = condition
            contract["lai_training_protocol"] = plan["protocol"]
            contract_id = family + "." + condition
            candidate_path = output_root / (contract_id + ".candidate.json")
            candidate = {
                "schema": CONTRACT_CANDIDATE_SCHEMA,
                "status": "PENDING_EXACT_DEVELOPMENT_ADMISSION",
                "contract_id": contract_id, "family": family,
                "condition": condition, "chromosome": plan["chromosome"],
                "resolved": resolved_spec,
                "fit_only": True, "optimizer_updates": 96,
                "fit_metadata_entries": 48, "select_metadata_entries": 48,
                "select_values_loaded": False, "contract_without_admission": contract,
            }
            write_fresh(candidate_path, candidate)
            contracts.append({"contract_id": contract_id,
                              "candidate": spec(candidate_path),
                              "resolved": resolved_spec})
    require(len(contracts) == 10, "exactly ten family-condition fits are required")
    value = {
        "schema": CONTRACT_PLAN_SCHEMA,
        "status": "TEN_FIT_ONLY_CONTRACTS_PENDING_EXACT_ADMISSIONS",
        "chromosome": plan["chromosome"],
        "request_plan": plan_spec, "materialization": materialization_spec,
        "contract_base": contract_base_spec, "optimizer_updates_per_fit": 96,
        "contracts": contracts, "select_metadata_bound": True,
        "select_values_loaded": False, "fit_only": True,
        "training_started": False, "evaluate_opened": False,
    }
    return write_fresh(output, value)


def _validate_contract_admissions_header(candidate_plan, candidate_plan_spec, admissions):
    expected = {row["contract_id"] for row in candidate_plan.get("contracts", [])}
    require(candidate_plan.get("schema") == CONTRACT_PLAN_SCHEMA
            and bool(normalize_chromosome(candidate_plan.get("chromosome")))
            and candidate_plan.get("select_values_loaded") is False
            and len(expected) == 10
            and admissions.get("schema") == CONTRACT_ADMISSIONS_SCHEMA
            and admissions.get("candidate_plan") == candidate_plan_spec
            and admissions.get("chromosome") == candidate_plan["chromosome"]
            and set(admissions.get("admissions", {})) == expected
            and admissions.get("select_values_loaded") is False
            and admissions.get("evaluate_opened") is False,
            "contract admissions do not bind the exact ten chromosome-specific candidates")
    return expected


def finalize_contracts(candidate_plan_path, candidate_plan_sha256,
                       admissions_path, admissions_sha256, output_root, output):
    candidate_plan_spec = spec(candidate_plan_path, candidate_plan_sha256)
    candidate_plan = campaign.read_json(candidate_plan_spec["path"],
                                        candidate_plan_spec["sha256"])
    admissions_spec = spec(admissions_path, admissions_sha256)
    admissions = campaign.read_json(admissions_spec["path"], admissions_spec["sha256"])
    _validate_contract_admissions_header(candidate_plan, candidate_plan_spec, admissions)
    from r02_context_train import validate_contract
    output_root = Path(output_root).resolve()
    rows = []
    for row in candidate_plan["contracts"]:
        candidate = campaign.read_json(
            campaign.descriptor(row["candidate"], "contract candidate"),
            row["candidate"]["sha256"])
        require(candidate.get("schema") == CONTRACT_CANDIDATE_SCHEMA
                and candidate.get("chromosome") == candidate_plan["chromosome"],
                "contract candidate differs from the chromosome-specific plan")
        admission_spec = admissions["admissions"][row["contract_id"]]
        admission_path = campaign.descriptor(admission_spec, "contract admission")
        admission_review = campaign.read_json(admission_path, admission_spec["sha256"])
        require(admission_review.get("status") == "APPROVED_DEVELOPMENT"
                and admission_review.get("scope") == "DEVELOPMENT"
                and admission_review.get("chromosome") == candidate_plan["chromosome"]
                and admission_review.get("evaluate_opened") is False,
                "contract admission is not an authenticated DEVELOPMENT approval")
        contract = candidate["contract_without_admission"]
        require(contract.get("training_authorized") is False,
                "candidate must remain unauthorized before its exact review")
        contract["training_authorized"] = True
        contract["admission"] = {
            "path": str(admission_path.resolve()),
            "sha256": admission_spec["sha256"],
            "status": admission_review["status"],
        }
        validate_contract(contract)
        path = output_root / (row["contract_id"] + ".contract.json")
        write_fresh(path, contract)
        rows.append({"contract_id": row["contract_id"], "contract": spec(path),
                     "resolved": row["resolved"], "fit_only": True})
    value = {
        "schema": CONTRACT_SET_SCHEMA,
        "status": "TEN_ADMITTED_FIT_ONLY_CONTRACTS_COMPLETE",
        "chromosome": candidate_plan["chromosome"],
        "candidate_plan": candidate_plan_spec, "admissions": admissions_spec,
        "contracts": rows, "optimizer_updates_per_fit": 96,
        "select_metadata_bound": True, "select_values_loaded": False,
        "fit_only": True, "training_started": False, "evaluate_opened": False,
    }
    return write_fresh(output, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in ("plan", "plan_sha256", "completion", "completion_sha256", "protocol",
                 "protocol_sha256", "prepared_root", "resolved_cnn",
                 "resolved_cnn_sha256", "resolved_attention",
                 "resolved_attention_sha256", "wrapper_root", "output"):
        plan.add_argument("--" + name.replace("_", "-"), required=True)
    create = commands.add_parser("materialize")
    for name in ("plan", "plan_sha256", "admission", "admission_sha256", "output"):
        create.add_argument("--" + name.replace("_", "-"), required=True)
    candidates = commands.add_parser("contract-candidates")
    for name in ("plan", "plan_sha256", "materialization", "materialization_sha256",
                 "contract_base", "contract_base_sha256", "output_root", "output"):
        candidates.add_argument("--" + name.replace("_", "-"), required=True)
    finalize = commands.add_parser("finalize-contracts")
    for name in ("candidate_plan", "candidate_plan_sha256", "admissions",
                 "admissions_sha256", "output_root", "output"):
        finalize.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = build(args.plan, args.plan_sha256, args.completion,
                      args.completion_sha256, args.protocol, args.protocol_sha256,
                      args.prepared_root, args.resolved_cnn, args.resolved_cnn_sha256,
                      args.resolved_attention, args.resolved_attention_sha256,
                      args.wrapper_root, args.output)
    elif args.command == "materialize":
        value = materialize(args.plan, args.plan_sha256, args.admission,
                            args.admission_sha256, args.output)
    elif args.command == "contract-candidates":
        value = contract_candidates(
            args.plan, args.plan_sha256, args.materialization,
            args.materialization_sha256, args.contract_base,
            args.contract_base_sha256, args.output_root, args.output)
    else:
        value = finalize_contracts(
            args.candidate_plan, args.candidate_plan_sha256, args.admissions,
            args.admissions_sha256, args.output_root, args.output)
    print(json.dumps(value, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
