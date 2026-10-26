#!/usr/bin/env python3
"""Audit existing LAI lineage; reciprocal protocols are dependent, never pooled.

This is an admission check, not an estimator of biological independence. Historical
population/IBD labels and sensitivity signals retain their original meanings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from r02_lai_target_dependencies import read_json, require, sha256

PROTOCOLS = {
    "FOLD_A_TO_FOLD_B": {"FOLD_A": "FIT", "FOLD_B": "SELECT"},
    "FOLD_B_TO_FOLD_A": {"FOLD_A": "SELECT", "FOLD_B": "FIT"},
}
DEPENDENCIES = (
    "full_scenario_universe", "independent_separation", "common_event_channels",
    "flare_baseline", "quantity_density_identity_controls",
    "ancestry_probability_boundary_metrics", "justified_search",
    "checkpoint_recovery", "measured_resources",
)
IDENTITIES = ("person", "donor", "family", "atomic_unit", "population", "root", "truth", "mosaic")
DEVELOPMENT_SCOPE_SCHEMA = "r02_lai_directional_development_scope_v1"
DEVELOPMENT_BATCH_SCHEMA = "r02_lai_directional_development_batches_v1"
DEVELOPMENT_CLAIM = "FIT_SELECT_DEVELOPMENT_CONDITIONAL_ON_RECORDED_DONORS_AND_SHARED_REFERENCE_PANEL_NO_RESERVED_PERFORMANCE"


def authenticated(spec):
    require(isinstance(spec, dict) and {"path", "sha256"} <= spec.keys(), "missing pinned evidence")
    path = Path(spec["path"])
    require(path.is_file() and sha256(path) == spec["sha256"], "evidence differs: " + str(path))
    return path


def rows(spec):
    with authenticated(spec).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def identities(spec):
    values = identity_values(spec)
    return set(values)


def identity_values(spec):
    """Read one authenticated, ordered identity axis."""
    values = authenticated(spec).read_text().splitlines()
    require(values and len(set(values)) == len(values) and all(values), "invalid identity list")
    require(spec.get("count", len(values)) == len(values), "identity count differs")
    return values


def axis_sha256(values):
    """Match the ordered string-axis digest written by the LAI channel adapter."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode() + b"\0")
    return digest.hexdigest()


def cross_roles(units):
    """Return exact collisions across roles, including renamed mosaics."""
    by_key = {key: defaultdict(lambda: defaultdict(set)) for key in IDENTITIES}
    for unit in units:
        for key in IDENTITIES:
            for identity in unit["identities"][key]:
                by_key[key][identity][unit["role"]].add(unit["case_id"])
    return {key: [{"identity": identity, "roles": {r: sorted(c) for r, c in role_map.items()}}
                  for identity, role_map in sorted(values.items()) if len(role_map) > 1]
            for key, values in by_key.items()}


def assignment_identity(assignments):
    """Hash case, source and person/component axes admitted for one direction."""
    digest = hashlib.sha256()
    for row in sorted((item["case_id"], item["protocol_id"], item["role"], item["component_id"],
                       item["source_manifest_sha256"], item["overlay_sha256"],
                       str(item["person_count"]), item["person_axis_sha256"],
                       item["component_axis_sha256"])
                      for item in assignments):
        digest.update(("\t".join(row) + "\n").encode())
    return digest.hexdigest()


def _development_batches(spec, protocol_id, assignments):
    """Authenticate complete FIT/SELECT neural batches without opening EVALUATE."""
    if spec is None:
        return None, ["AUTHENTICATED_FIT_SELECT_CHANNEL_BATCHES_ABSENT"]
    manifest_path = authenticated(spec)
    manifest = read_json(manifest_path)
    require(set(manifest) == {"schema", "protocol_id", "evaluate_opened", "tasks"}
            and manifest["schema"] == DEVELOPMENT_BATCH_SCHEMA,
            "invalid directional development batch manifest")
    require(manifest["protocol_id"] == protocol_id, "batch manifest crosses reciprocal protocols")
    require(manifest["evaluate_opened"] is False, "development batches opened EVALUATE")
    expected_rows = {(row["case_id"], row["role"]): row for row in assignments}
    expected = set(expected_rows)
    observed, records = set(), []
    for task in manifest["tasks"]:
        require(set(task) == {"case_id", "protocol_id", "role", "batch", "receipt", "input_request"},
                "invalid directional development batch task")
        require(task["protocol_id"] == protocol_id and task["role"] in {"FIT", "SELECT"},
                "batch task has unknown role or crosses reciprocal protocols")
        key = (task["case_id"], task["role"])
        require(key in expected and key not in observed, "batch task is unexpected or duplicated")
        assignment = expected_rows[key]
        batch_path = authenticated(task["batch"])
        receipt = read_json(authenticated(task["receipt"]))
        request_path = authenticated(task["input_request"])
        from r02_lai_neural_input import audit_request
        input_admission = audit_request(request_path, task["input_request"]["sha256"])
        input_request = read_json(request_path)
        require(receipt.get("schema") == "r02_lai_neural_input_batch_v1"
                and receipt.get("status") == "ASSEMBLED_NOT_TRAINED_OR_EVALUATED",
                "LAI channel batch receipt has wrong schema or stage")
        require(receipt.get("batch_sha256") == task["batch"]["sha256"] == sha256(batch_path),
                "LAI channel receipt does not authenticate batch")
        require(receipt.get("case_id") == task["case_id"]
                and receipt.get("direction") == protocol_id
                and receipt.get("expected_role") == task["role"],
                "LAI channel batch case, direction, or role differs")
        require(receipt.get("request_sha256") == task["input_request"]["sha256"]
                and input_admission.get("status") == "READY_FOR_BOUNDED_ASSEMBLY"
                and input_admission.get("case_id") == task["case_id"]
                and input_admission.get("direction") == protocol_id
                and input_admission.get("expected_role") == task["role"],
                "assembled batch is not bound to an admitted input request")
        material = read_json(authenticated(input_request["inputs"]["material_manifest"]))
        channel = read_json(authenticated(input_request["inputs"]["channel_receipt"]))
        require(material.get("source_manifest_sha256") == assignment["source_manifest_sha256"]
                and input_request["inputs"]["overlay"]["sha256"] == assignment["overlay_sha256"]
                and channel.get("person_axis_sha256") == assignment["person_axis_sha256"],
                "input request does not originate from the audited source case")
        # Load only the three compact identity arrays; genomic feature arrays stay unopened.
        import numpy as np
        with np.load(batch_path, allow_pickle=False) as source:
            require({"person_id", "component_id", "role"} <= set(source.files),
                    "assembled batch lacks identity axes")
            people = source["person_id"].astype(str).tolist()
            components = source["component_id"].astype(str).tolist()
            roles = source["role"].astype(str).tolist()
        require(len(people) == assignment["person_count"]
                and len(set(people)) == len(people)
                and axis_sha256(people) == assignment["person_axis_sha256"],
                "assembled batch person identities differ from audited source case")
        require(len(components) == assignment["person_count"] == len(people),
                "assembled batch component identity count differs from person count")
        require(axis_sha256(components) == assignment["component_axis_sha256"]
                and set(components) == {assignment["component_id"]},
                "assembled batch component identities differ from audited source case")
        require(set(roles) == {task["role"]} and len(roles) == len(people),
                "assembled batch role axis differs from audited source case")
        require(receipt.get("evaluate_opened") is False
                and receipt.get("training_authorized") is False
                and receipt.get("model_selection_authorized") is False,
                "assembled batch exceeded its data-preparation scope")
        observed.add(key)
        records.append({"case_id": task["case_id"], "protocol_id": protocol_id,
                        "role": task["role"], "batch_sha256": task["batch"]["sha256"],
                        "receipt_sha256": task["receipt"]["sha256"],
                        "input_request_sha256": task["input_request"]["sha256"],
                        "person_axis_sha256": assignment["person_axis_sha256"],
                        "component_axis_sha256": assignment["component_axis_sha256"]})
    require(observed == expected, "directional batches do not cover every FIT/SELECT assignment")
    require({role for _, role in observed} == {"FIT", "SELECT"}, "both development roles require batches")
    return {"manifest": spec, "manifest_sha256": sha256(manifest_path),
            "tasks": sorted(records, key=lambda row: (row["role"], row["case_id"]))}, []


def audit_directional_development(result, scope):
    """Separate a single development direction without claiming final evaluation.

    This function narrows the strict audit. It never converts the reciprocal
    directions into independent repetitions and never treats absent EVALUATE
    metadata as observed reserved performance.
    """
    required = {"schema", "protocol_id", "development_claim", "actual_uses",
                "reserve_protection", "reference_dependency", "batch_manifest"}
    require(isinstance(scope, dict) and set(scope) == required
            and scope["schema"] == DEVELOPMENT_SCOPE_SCHEMA,
            "invalid explicit LAI directional development scope")
    protocol_id = scope["protocol_id"]
    require(protocol_id in PROTOCOLS, "unknown reciprocal development protocol")
    require(scope["development_claim"] == DEVELOPMENT_CLAIM,
            "conditional development claim was not acknowledged")
    assignments = [row for row in result["assignments"] if row["protocol_id"] == protocol_id]
    require(assignments and {row["role"] for row in assignments} == {"FIT", "SELECT"},
            "direction lacks explicit FIT and SELECT assignments")
    counts = {role: sum(row["role"] == role for row in assignments) for role in ("FIT", "SELECT")}
    uses = scope["actual_uses"]
    expected_uses = {
        "protocol_id": protocol_id,
        "assignment_identity_sha256": assignment_identity(assignments),
        "query_units": len(assignments),
        "role_query_units": counts,
        "donor_role": "SOURCE_VALID_GENERATES_SIMULATED_MOSAICS",
        "reference_role": "REF_TRAIN_SHARED_FIXED_PANEL",
        "reciprocal_protocol_used": False,
        "source_test_used": False,
        "evaluate_used": False,
    }
    require(uses == expected_uses, "actual-use ledger differs from audited directional assignments")
    reserve = scope["reserve_protection"]
    require(reserve == {
        "evaluate_metadata_used": False,
        "evaluate_responses_opened": False,
        "source_test_opened": False,
        "future_nam_reserve_status": "NOT_IDENTIFIED_IN_CURRENT_INVENTORY",
        "future_nam_reserve_used": False,
        "reserved_performance_claimed": False,
    }, "future reserve protection is missing or overclaims an existing reserve")
    reference = scope["reference_dependency"]
    require(set(reference) == {"panel_role", "panel_reused", "flagged_pair_count",
                               "kinship_sensitivity_status", "biological_independence_claimed",
                               "recorded_cross_role_components", "affected_source_donors",
                               "affected_references", "exclusions"},
            "invalid reference dependency ledger")
    sensitivity = result.get("kinship_dependency_sensitivity")
    require(isinstance(sensitivity, dict) and sensitivity.get("thresholds"),
            "reference-family sensitivity evidence is absent")
    lowest_phi = min(sensitivity["thresholds"])
    scenarios = [row for row in sensitivity.get("scenarios", [])
                 if row.get("phi") == lowest_phi and row.get("pcs_rule") == "ANY_RECORDED_PC"]
    require(len(scenarios) == 1, "reference-family sensitivity scenario is absent or ambiguous")
    components = scenarios[0].get("cross_role_components", [])
    affected_source = sum(sum(value for role, value in row.get("role_counts", {}).items()
                              if role in {"FOLD_A", "FOLD_B"}) for row in components)
    affected_reference = sum(row.get("role_counts", {}).get("REF_TRAIN", 0) for row in components)
    require(reference["panel_role"] == "REF_TRAIN"
            and reference["panel_reused"] is True
            and reference["flagged_pair_count"] == result["reference_donor_flagged_pairs"]
            and reference["kinship_sensitivity_status"] == "UNRESOLVED"
            and reference["recorded_cross_role_components"] == len(components)
            and reference["affected_source_donors"] == affected_source
            and reference["affected_references"] == affected_reference
            and reference["biological_independence_claimed"] is False,
            "shared reference-panel dependency differs or was hidden")
    require(isinstance(reference["exclusions"], list), "reference exclusions must be an explicit list")
    for exclusion in reference["exclusions"]:
        authenticated(exclusion)

    protocol = result["protocols"][protocol_id]
    collisions = protocol["collision_counts"]
    require(set(collisions) == set(IDENTITIES), "directional collision axes are incomplete")
    incomplete = protocol.get("incomplete_identity_counts")
    require(isinstance(incomplete, dict) and set(incomplete) == set(IDENTITIES),
            "directional identity-integrity axes are incomplete")
    blockers = ["CROSS_ROLE_COLLISION:" + protocol_id] if any(collisions.values()) else []
    blockers += ["IDENTITY_AXIS_INCOMPLETE:" + protocol_id + ":" + key
                 for key, count in incomplete.items() if count]
    batches, batch_blockers = _development_batches(scope["batch_manifest"], protocol_id, assignments)
    blockers += batch_blockers
    separation_passed = not any(collisions.values()) and not any(incomplete.values())
    channels_complete = batches is not None
    return {
        "schema": "r02_lai_directional_development_audit_v1",
        "status": ("DIRECTIONAL_FIT_SELECT_BATCHES_AUTHENTICATED_EVALUATE_CLOSED"
                   if separation_passed and channels_complete else
                   "DIRECTIONAL_SEPARATION_AUDITED_INPUTS_INCOMPLETE"),
        "protocol_id": protocol_id,
        "development_claim": DEVELOPMENT_CLAIM,
        "development_separation_passed": separation_passed,
        "development_training_separation_allowed": separation_passed and channels_complete,
        "training_allowed": separation_passed and channels_complete,
        "evaluate_opened": False,
        "evaluate_metadata_present": False,
        "reserved_performance_claimed": False,
        "pooling_allowed": False,
        "cross_protocol_selection_allowed": False,
        "independent_replicates": False,
        "role_query_units": counts,
        "assignment_identity_sha256": assignment_identity(assignments),
        "collision_counts": collisions,
        "incomplete_identity_counts": incomplete,
        "actual_uses": uses,
        "reserve_protection": reserve,
        "reference_dependency": reference,
        "authenticated_batches": batches,
        "blockers": sorted(set(blockers)),
        "remaining_scientific_dependencies": result["scientific_dependencies"],
        "allowed_claim": "DEVELOPMENT_ONLY_CONDITIONAL_ON_RECORDED_DONORS_AND_SHARED_REFERENCE_PANEL",
        "forbidden_claims": ["reserved performance", "independent reciprocal replication",
                             "biological independence", "generalization to future NAM donors"],
    }


def audit(request_path, expected_sha=None):
    if expected_sha:
        require(sha256(request_path) == expected_sha, "audit request differs")
    request = read_json(request_path)
    require(request.get("schema") == "r02_lai_separation_request_v1", "unsupported separation request")
    require(request.get("policy") == "SEPARATE_DEPENDENT_RECIPROCAL_PROTOCOLS", "protocol policy differs")
    require(request.get("evaluate_opened") is False, "EVALUATE must remain closed")
    require(request.get("cross_protocol_selection_allowed") is False, "cross-protocol selection forbidden")
    sources = request["sources"]
    inventory = read_json(authenticated(sources["units"]))
    design = read_json(authenticated(sources["design_manifest"]))
    roster_rows = rows(sources["roles"])
    original_rows = rows(sources["original_roles"])
    require(len({r["sample_id"] for r in roster_rows}) == len(roster_rows), "duplicate donor roster identity")
    require(len({r["sample_id"] for r in original_rows}) == len(original_rows), "duplicate original identity")
    roster = {r["sample_id"]: r for r in roster_rows}
    original = {r["sample_id"]: r for r in original_rows}
    require(design["outputs_sha256"]["roles.private.tsv"] == sources["roles"]["sha256"], "unbound donor roster")
    require(design["inputs"]["original_roles"]["sha256"] == sources["original_roles"]["sha256"], "unbound original units")
    require(design["outputs_sha256"]["kinship_checks.json"] == sources["kinship_checks"]["sha256"], "unbound kinship sensitivity")
    authenticated(sources["kinship_checks"])
    blocked, units, assignments = [], [], []
    seen = set()
    for item in inventory["units"]:
        case, direction, fold = item["case_id"], item["direction"], item["fold"]
        require(direction in PROTOCOLS and fold in PROTOCOLS[direction], "unknown reciprocal protocol")
        require(item["role"] == PROTOCOLS[direction][fold], "role differs from frozen protocol")
        require((case, direction) not in seen, "duplicate query unit")
        seen.add((case, direction))
        person_values = identity_values(item["sample_ids"])
        people, donors = set(person_values), identities(item["donor_ids"])
        for donor in donors:
            require(donor in roster and donor in original, "unmapped donor identity")
            require(roster[donor]["role"] == original[donor]["role"] == "SOURCE_VALID", "donor role changed")
            require(roster[donor]["split"] == fold, "generating donor crosses source fold")
        # Authenticate small primary receipts; array hashes are inventoried separately.
        for name in ("source_manifest", "material_manifest", "truth", "validated_split", "overlay_receipt", "request"):
            authenticated(item["evidence"][name])
        source = read_json(item["evidence"]["source_manifest"]["path"])
        require(item["source_manifest_sha256"] == item["evidence"]["source_manifest"]["sha256"], "source root differs")
        require(item["truth_sha256"] == item["evidence"]["truth"]["sha256"] == source["outputs_sha256"]["truth.private.tsv.gz"], "truth lineage differs")
        split = read_json(item["evidence"]["validated_split"]["path"])
        match = [r for r in split["assignments"] if r["case_id"] == case]
        require(len(match) == 1 and match[0]["role"] == item["role"] and match[0]["component_id"] == item["component_id"], "component lineage differs")
        if "mosaic_hashes" not in item:
            blocked.append("MOSAIC_IDENTITY_NOT_VERIFIED")
        mosaic_rows = rows(item["mosaic_hashes"]) if "mosaic_hashes" in item else []
        if mosaic_rows:
            require(len(mosaic_rows) == len(people) and {r["sample_id"] for r in mosaic_rows} == people,
                    "mosaic and person axes differ")
        mosaics = {r["canonical_mosaic_sha256"] for r in mosaic_rows}
        unit = {"case_id": case, "protocol_id": direction, "role": item["role"], "identities": {
            "person": people, "donor": donors,
            "family": {roster[d]["family_component"] for d in donors},
            "atomic_unit": {original[d]["atomic_unit_id"] for d in donors},
            "population": {original[d]["canonical_population"] for d in donors},
            "root": {item["source_manifest_sha256"]},
            "truth": {item["canonical_truth_sha256"]}, "mosaic": mosaics,
        }}
        require(all(unit["identities"][key] and "" not in unit["identities"][key] for key in IDENTITIES if key != "mosaic"), "unknown separation identity")
        units.append(unit)
        assignments.append({"case_id": case, "protocol_id": direction, "role": item["role"],
                            "component_id": item["component_id"], "overlay_sha256": item["overlay_sha256"],
                            "source_manifest_sha256": item["source_manifest_sha256"],
                            "person_count": len(person_values),
                            "person_axis_sha256": axis_sha256(person_values),
                            "component_axis_sha256": axis_sha256(
                                [item["component_id"]] * len(person_values))})
    require(len(units) == inventory["unit_count"] == request["expected_query_units"], "incomplete query universe")
    protocols = {}
    for protocol in PROTOCOLS:
        subset = [u for u in units if u["protocol_id"] == protocol]
        require({u["role"] for u in subset} == {"FIT", "SELECT"}, "both development roles required")
        collisions = cross_roles(subset)
        incomplete = {key: sum(not unit["identities"][key] for unit in subset)
                      for key in IDENTITIES}
        if any(collisions.values()): blocked.append("CROSS_ROLE_COLLISION:" + protocol)
        protocols[protocol] = {"collision_counts": {k: len(v) for k, v in collisions.items()},
                               "collisions": collisions,
                               "incomplete_identity_counts": incomplete,
                               "identity_integrity_complete": not any(incomplete.values()),
                               "query_units": len(subset)}
    pooled = cross_roles(units)
    blocked += ["INDEPENDENT_EVALUATE_COMPONENT_ABSENT", "FULL_SCIENTIFIC_DEPENDENCIES_NOT_VERIFIED"]
    summary = design["summary"]
    if summary["cross_reference_donor_flagged_pairs_at_lowest_phi"]:
        blocked.append("REFERENCE_DONOR_KINSHIP_SENSITIVITY_UNRESOLVED")
    if not summary["independence_certified"]:
        blocked.append("BIOLOGICAL_INDEPENDENCE_NOT_CERTIFIED")
    kinship_dependencies = None
    if 'pair_review_manifest' in sources:
        from r02_lai_kinship_dependencies import assess
        review = read_json(authenticated(sources['pair_review_manifest']))
        require(design['inputs']['pair_review']['sha256'] == sources['pair_review_manifest']['sha256'],
                'pair review not bound to original design')
        require(review['outputs_sha256']['pair_review.private.tsv'] == sources['pair_review']['sha256'],
                'pair table not bound to review')
        kinship_dependencies = assess(roster_rows, original_rows, rows(sources['pair_review']),
                                     read_json(authenticated(sources['kinship_checks'])))
        if 'ibd_manifest' in sources:
            from r02_lai_kinship_dependencies import link_existing_ibd
            ibd = read_json(authenticated(sources['ibd_manifest']))
            for key, filename in [('ibd_pairs', 'pairs.private.tsv'), ('ibd_summary', 'summary.json')]:
                require(ibd['outputs_sha256'][filename] == sources[key]['sha256'], 'unbound IBD evidence')
            kinship_dependencies['existing_ibd'] = link_existing_ibd(
                kinship_dependencies, rows(sources['ibd_pairs']), read_json(authenticated(sources['ibd_summary'])))
            blocked.append('IBD_DETECTABILITY_NOT_CALIBRATED')
    else:
        blocked.append('PAIR_LEVEL_KINSHIP_DEPENDENCIES_NOT_TRACED')
    return {"schema": "r02_lai_separation_audit_v1", "status": "AUDITED_DEVELOPMENT_PROTOCOLS_PRODUCTION_BLOCKED",
            "request_sha256": sha256(request_path), "assignments": assignments, "protocols": protocols,
            "pooled_collision_counts": {k: len(v) for k, v in pooled.items()},
            "pooling_allowed": False, "cross_protocol_selection_allowed": False,
            "independent_replicates": False, "evaluate_opened": False, "evaluate_assignments": [],
            "training_allowed": False, "blockers": sorted(set(blocked)),
            "scientific_dependencies": {key: "NOT_VERIFIED_FOR_PRODUCTION" for key in DEPENDENCIES},
            "reference_donor_flagged_pairs": summary["cross_reference_donor_flagged_pairs_at_lowest_phi"],
            "kinship_dependency_sensitivity": kinship_dependencies,
            "allowed_use": "PINNED_INPUT_REUSE_WITHIN_ONE_DEPENDENT_DEVELOPMENT_PROTOCOL",
            "limits": ["Nominal population and component disjointness is not biological independence.",
                       "Content identity denotes supplied simulated mosaics, not validated parental phase.",
                       "A fixed external reference panel does not resolve its kinship sensitivity."]}


def require_single_protocol(requests):
    """Used before Nextflow stages any neural input, including cached tasks."""
    directions = {r.get("protocol_id") for r in requests}
    require(len(directions) == 1 and next(iter(directions), None) in PROTOCOLS, "one explicit LAI protocol required; pooling is forbidden")
    direction = next(iter(directions))
    require(all(r.get("direction") == direction for r in requests), "protocol and direction differ")
    return direction


def require_contract_directional_batches(contract, directional):
    """Bind trainer entries to the exact authenticated cases, roles and direction."""
    protocol_id = directional["protocol_id"]
    batches = contract.get("batches")
    require(isinstance(batches, dict) and set(batches) == {"FIT", "SELECT"}
            and all(isinstance(rows, list) and rows for rows in batches.values()),
            "LAI development requires nonempty FIT and SELECT batch lists")
    authenticated_batches = directional.get("authenticated_batches")
    require(isinstance(authenticated_batches, dict),
            "LAI development has no authenticated FIT/SELECT channel batches")
    admitted = {(row["case_id"], row["role"], row["batch_sha256"], row["receipt_sha256"],
                 row["input_request_sha256"])
                for row in authenticated_batches["tasks"]}
    supplied = set()
    for role, entries in batches.items():
        for entry in entries:
            require(entry.get("protocol_id") == protocol_id and entry.get("role") == role,
                    "training batch crosses protocol or has missing/unknown role")
            require(all(entry.get(key) for key in
                        ("lai_case_id", "sha256", "receipt_path", "receipt_sha256",
                         "input_request_sha256")),
                    "training batch lacks directional provenance")
            key = (entry["lai_case_id"], role, entry["sha256"], entry["receipt_sha256"],
                   entry["input_request_sha256"])
            require(key not in supplied, "duplicate training batch entry")
            supplied.add(key)
    require(supplied == admitted, "training batches differ from authenticated directional batch manifest")
    return protocol_id


def require_training_separation(contract):
    """Require one complete directional FIT/SELECT scope for development training."""
    spec = contract.get("lai_separation_request")
    require(spec is not None, "LAI production requires authenticated joint separation and scientific admission")
    result = audit(authenticated(spec), spec["sha256"])
    scope_spec = contract.get("lai_development_scope")
    require(isinstance(scope_spec, dict), "LAI development requires a pinned directional scope")
    scope = read_json(authenticated(scope_spec))
    if scope.get("schema") == "r02_lai_factorized_development_scope_v1":
        chromosome = str(contract.get("lai_chromosome", ""))
        require(chromosome.isdigit() and 1 <= int(chromosome) <= 22
                and chromosome == str(int(chromosome)),
                "factorized LAI contract requires one canonical autosome")
        require(set(scope) == {"schema", "protocol_id", "chromosome", "cases",
                               "source_directional_audit", "shared_reference_dependency",
                               "select_values_loaded", "evaluate_opened"}
                and scope["protocol_id"] == "FOLD_A_TO_FOLD_B"
                and scope["chromosome"] == chromosome
                and scope["select_values_loaded"] is False
                and scope["evaluate_opened"] is False,
                "invalid factorized protocol-chromosome directional scope")
        source_audit = read_json(authenticated(scope["source_directional_audit"]))
        require(source_audit.get("protocol_id") == "FOLD_A_TO_FOLD_B"
                and source_audit.get("development_separation_passed") is True
                and source_audit.get("evaluate_opened") is False
                and not any(source_audit.get("collision_counts", {}).values())
                and scope["shared_reference_dependency"] ==
                    source_audit.get("reference_dependency"),
                "source directional audit does not support this dependent FIT/SELECT scope")
        cases = scope["cases"]
        require(isinstance(cases, dict) and set(cases) == {"FIT", "SELECT"}
                and all(len(values) == 3 and len(set(values)) == 3
                        for values in cases.values()),
                "factorized scope requires three cases per development role")
        observed = {role: {entry.get("lai_case_id") for entry in entries}
                    for role, entries in contract.get("batches", {}).items()}
        from r02_lai_neural_input import parse_case
        observed_chromosomes = {
            parse_case(entry.get("lai_case_id"))[1]
            for entries in contract.get("batches", {}).values() for entry in entries
        }
        require(observed == {role: set(values) for role, values in cases.items()}
                and observed_chromosomes == {chromosome}
                and all(entry.get("protocol_id") == "FOLD_A_TO_FOLD_B"
                        and entry.get("role") == role
                        for role, entries in contract["batches"].items()
                        for entry in entries),
                "contract batches differ from the factorized directional cases or chromosome")
        return {"protocol_id": "FOLD_A_TO_FOLD_B",
                "chromosome": chromosome,
                "development_training_separation_allowed": True,
                "independent_replicates": False,
                "shared_reference_dependency": scope["shared_reference_dependency"]}
    directional = audit_directional_development(result, scope)
    require(directional["development_training_separation_allowed"],
            "LAI development blocked: " + "; ".join(directional["blockers"]))
    require_contract_directional_batches(contract, directional)
    return directional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.request, args.request_sha256)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    if output.exists():
        require(output.read_text() == payload, "existing audit differs; preserve and use a new revision")
    else:
        with output.open("x") as stream: stream.write(payload)
    print(json.dumps({"status": result["status"], "training_allowed": result["training_allowed"]}))


if __name__ == "__main__":
    main()
