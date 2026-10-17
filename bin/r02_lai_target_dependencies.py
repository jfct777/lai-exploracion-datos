#!/usr/bin/env python3
"""Resolve LAI target dependencies without assigning a training split.

The current 12 simulated cases do not retain donor-per-mosaic provenance.  In
that situation this module deliberately joins cases within each authenticated
SOURCE_VALID donor fold. Disjoint donor-fold family components remain separate;
the shared REF_TRAIN predictor panel does not imply shared target generators. A
finer graph is accepted only from an explicit donor-usage file; local-ancestry
truth is never read or accepted as a feature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPORT_SCHEMA = "r02_lai_target_dependency_report_v1"
SPLIT_SCHEMA = "r02_lai_explicit_case_split_v1"
DONOR_USAGE_SCHEMA = "r02_lai_mosaic_donor_usage_v1"
ROLES = {"FIT", "SELECT", "TEST"}
CASE_RE = re.compile(
    r"^(?P<batch>[^.]+)\.(?P<scenario>[^.]+)\.chr(?P<chromosome>[0-9]+)\."
    r"(?P<donor_fold>FOLD_[AB])\.r(?P<replicate>[1-9][0-9]*)$"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(
        Path(path).read_text(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-finite JSON: " + value)
        ),
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


class Components:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def parse_cases(requirements):
    require(requirements.get("schema") == "r02_lai_context_requirements_v1",
            "unsupported requirements schema")
    cases = requirements.get("cases")
    require(isinstance(cases, list) and len(cases) == 12, "exactly 12 cases required")
    parsed = {}
    for case in cases:
        require(isinstance(case, dict), "case must be an object")
        case_id = case.get("case_id")
        match = CASE_RE.fullmatch(case_id or "")
        require(match is not None and case_id not in parsed, "invalid or duplicate case_id")
        fields = match.groupdict()
        require(str(case.get("chromosome")) == fields["chromosome"] and
                case.get("donor_fold") == fields["donor_fold"],
                "case_id chromosome/fold differs from case fields")
        require(type(case.get("people")) is int and case["people"] > 0,
                "positive people count required")
        parsed[case_id] = {
            "case_id": case_id,
            "scenario": fields["scenario"],
            "chromosome": fields["chromosome"],
            "donor_fold": fields["donor_fold"],
            "replicate": int(fields["replicate"]),
            "people": case["people"],
            "material_path": case.get("material_path"),
            "material_manifest_sha256": case.get("material_manifest_sha256"),
        }
    return parsed


def authenticate_design(design_manifest_path, scenarios_path, roles_path):
    manifest = read_json(design_manifest_path)
    require(manifest.get("schema") == "r02_lai_development_design_v1" and
            manifest.get("status") == "DEVELOPMENT_DESIGN_PREPARED_NOT_SIMULATION_AUTHORIZATION",
            "unsupported development design")
    expected = manifest.get("outputs_sha256", {})
    require(expected.get("scenarios.json") == sha256(scenarios_path) and
            expected.get("roles.private.tsv") == sha256(roles_path),
            "design outputs differ from authenticated manifest")
    scenarios = read_json(scenarios_path)
    require(scenarios.get("schema") == "r02_lai_demography_templates_v1" and
            scenarios.get("scope") == "CONDITIONAL_DEVELOPMENT_ONLY",
            "unsupported scenario design")
    reciprocal = scenarios.get("reciprocal_folds")
    require(isinstance(reciprocal, list) and reciprocal and
            all(x.get("confirmatory_test") is None for x in reciprocal),
            "design unexpectedly claims a confirmatory test")

    counts = Counter()
    components = defaultdict(set)
    with Path(roles_path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == ["sample_id", "role", "ancestry", "split", "family_component"],
                "unexpected private role columns")
        seen = set()
        for row in reader:
            require(row["sample_id"] and row["sample_id"] not in seen,
                    "empty or duplicate private panel identity")
            seen.add(row["sample_id"])
            require(row["role"] in {"REF_TRAIN", "SOURCE_VALID"} and row["family_component"],
                    "unsupported panel role/component")
            if row["role"] == "REF_TRAIN":
                require(row["split"] == "REFERENCE", "reference split differs")
            else:
                require(row["split"] in {"FOLD_A", "FOLD_B"}, "donor fold differs")
            counts[(row["role"], row["split"])] += 1
            components[(row["role"], row["split"])].add(row["family_component"])
    require(counts[("REF_TRAIN", "REFERENCE")] > 0 and
            all(counts[("SOURCE_VALID", fold)] > 0 for fold in ("FOLD_A", "FOLD_B")),
            "design lacks the shared reference panel or a donor fold")
    source_intersection = (components[("SOURCE_VALID", "FOLD_A")] &
                           components[("SOURCE_VALID", "FOLD_B")])
    require(not source_intersection,
            "SOURCE_VALID family component crosses authenticated donor folds")
    return {
        "design_manifest_sha256": sha256(design_manifest_path),
        "scenarios_sha256": sha256(scenarios_path),
        "roles_sha256": sha256(roles_path),
        "shared_reference_people": counts[("REF_TRAIN", "REFERENCE")],
        "source_valid_people_by_fold": {
            fold: counts[("SOURCE_VALID", fold)] for fold in ("FOLD_A", "FOLD_B")
        },
        "family_component_counts": {
            "REFERENCE": len(components[("REF_TRAIN", "REFERENCE")]),
            **{fold: len(components[("SOURCE_VALID", fold)]) for fold in ("FOLD_A", "FOLD_B")},
        },
        "source_valid_cross_fold_component_intersection_count": len(source_intersection),
        "source_valid_folds_component_disjoint": True,
        "confirmatory_test_defined": False,
    }


def authenticate_materials(cases):
    counts = Counter()
    for case_id, case in cases.items():
        path = Path(case["material_path"] or "") / "manifest.json"
        require(path.is_file() and sha256(path) == case["material_manifest_sha256"],
                "material manifest missing or changed: " + case_id)
        manifest = read_json(path)
        require(manifest.get("schema") == "r02_lai_genotype_material_v1" and
                manifest.get("case_id") == case_id and
                manifest.get("donor_split") == case["donor_fold"],
                "material identity/fold differs: " + case_id)
        predictors = set(manifest.get("predictor_files", []))
        targets = set(manifest.get("separate_target_files", []))
        require(predictors == {"alt_dosage.npy", "genotype_observed.npy", "loci.tsv"} and
                targets == {"truth_state.npy", "observed_mask.npy"} and
                predictors.isdisjoint(targets),
                "predictor/target separation differs: " + case_id)
        require(manifest.get("train_select_test_split_assigned") is False and
                manifest.get("independent_evaluation") is False,
                "material unexpectedly claims a split or independent evaluation")
        counts[case["donor_fold"]] += 1
    return {
        "authenticated_cases": len(cases),
        "cases_by_donor_fold": dict(sorted(counts.items())),
        "predictor_files": ["alt_dosage.npy", "genotype_observed.npy", "loci.tsv"],
        "target_only_files": ["truth_state.npy", "observed_mask.npy"],
        "truth_used_to_build_dependencies": False,
        "truth_admitted_as_predictor": False,
    }


def exact_components(cases, donor_usage_path):
    usage = read_json(donor_usage_path)
    require(usage.get("schema") == DONOR_USAGE_SCHEMA, "unsupported donor usage schema")
    records = usage.get("records")
    require(isinstance(records, list) and records, "donor usage records required")
    graph = Components(cases)
    donors_to_cases = defaultdict(set)
    donors_to_folds = defaultdict(set)
    targets_by_case = defaultdict(set)
    allowed = {"case_id", "target_id", "simulation_role", "donor_ids"}
    for record in records:
        require(isinstance(record, dict) and set(record) == allowed,
                "donor usage record has unexpected fields; truth/labels are forbidden")
        case_id = record["case_id"]
        target_id = record["target_id"]
        donors = record["donor_ids"]
        require(case_id in cases and record["simulation_role"] == "SIMULATED_TARGET",
                "unknown case or unsupported simulation role")
        require(isinstance(target_id, str) and target_id and target_id not in targets_by_case[case_id],
                "empty or duplicate target within case")
        require(isinstance(donors, list) and donors and len(donors) == len(set(donors)) and
                all(isinstance(value, str) and value for value in donors),
                "non-empty unique donor_ids required")
        targets_by_case[case_id].add(target_id)
        for donor in donors:
            donors_to_cases[donor].add(case_id)
            donors_to_folds[donor].add(cases[case_id]["donor_fold"])
    require(set(targets_by_case) == set(cases), "donor usage must cover all 12 cases")
    require(all(len(targets_by_case[case_id]) == cases[case_id]["people"] for case_id in cases),
            "donor usage must cover every target mosaic in each case")
    require(all(len(folds) == 1 for folds in donors_to_folds.values()),
            "donor token crosses authenticated disjoint SOURCE_VALID folds")
    for linked in donors_to_cases.values():
        linked = sorted(linked)
        for case_id in linked[1:]:
            graph.union(linked[0], case_id)
    return graph, {
        "mode": "EXACT_SHARED_DONORS",
        "donor_usage_sha256": sha256(donor_usage_path),
        "donor_usage_records": len(records),
        "distinct_donor_tokens": len(donors_to_cases),
        "private_donor_tokens_emitted": False,
        "limitation": "Components express observed donor reuse; they do not create biological independence.",
    }


def fallback_components(cases, design):
    graph = Components(cases)
    require(design.get("source_valid_folds_component_disjoint") is True and
            design.get("source_valid_cross_fold_component_intersection_count") == 0,
            "SOURCE_VALID fold disjunction is not authenticated")
    by_fold = defaultdict(list)
    for case_id, case in cases.items():
        by_fold[case["donor_fold"]].append(case_id)
    require(set(by_fold) == {"FOLD_A", "FOLD_B"}, "expected both donor folds")
    for members in by_fold.values():
        members = sorted(members)
        for case_id in members[1:]:
            graph.union(members[0], case_id)
    return graph, {
        "mode": "CONSERVATIVE_SOURCE_VALID_FOLD_COMPONENTS",
        "donor_usage_sha256": None,
        "donor_usage_records": 0,
        "distinct_donor_tokens": None,
        "private_donor_tokens_emitted": False,
        "limitation": (
            "No donor-per-mosaic provenance is present. Cases within each SOURCE_VALID fold are "
            "one conservative component because they draw from the same donor fold. FOLD_A and "
            "FOLD_B remain separate because their authenticated family components are disjoint. "
            "The shared REF_TRAIN predictor panel is not treated as a target-generating donor."
        ),
    }


def build_report(requirements_path, design_manifest_path, scenarios_path, roles_path,
                 donor_usage_path=None):
    requirements = read_json(requirements_path)
    cases = parse_cases(requirements)
    design = authenticate_design(design_manifest_path, scenarios_path, roles_path)
    materials = authenticate_materials(cases)
    scenario_ids = {case["scenario"] for case in cases.values()}
    scenario_design = read_json(scenarios_path)
    known = {item["scenario_id"] for item in scenario_design["scenarios"]}
    require(scenario_ids <= known, "case scenario absent from authenticated design")
    graph, provenance = (exact_components(cases, donor_usage_path) if donor_usage_path
                         else fallback_components(cases, design))
    grouped = defaultdict(list)
    for case_id in sorted(cases):
        grouped[graph.find(case_id)].append(case_id)
    components = []
    case_to_component = {}
    for members in sorted(grouped.values(), key=lambda value: value[0]):
        component_id = "lai-target-component-" + hashlib.sha256(
            "\n".join(members).encode()
        ).hexdigest()[:16]
        for case_id in members:
            case_to_component[case_id] = component_id
        components.append({
            "component_id": component_id,
            "case_ids": members,
            "case_count": len(members),
            "chromosomes": sorted({cases[x]["chromosome"] for x in members}, key=int),
            "donor_folds": sorted({cases[x]["donor_fold"] for x in members}),
            "replicates": sorted({cases[x]["replicate"] for x in members}),
            "simulated_people": sum(cases[x]["people"] for x in members),
        })
    return {
        "schema": REPORT_SCHEMA,
        "status": "DEPENDENCY_COMPONENTS_RESOLVED_EXPLICIT_SPLIT_REQUIRED",
        "requirements_sha256": sha256(requirements_path),
        "case_count": len(cases),
        "component_count": len(components),
        "components": components,
        "case_to_component": case_to_component,
        "dependency_provenance": provenance,
        "authenticated_design": design,
        "material_separation": materials,
        "split_policy": {
            "assignment_created": False,
            "allowed_roles": sorted(ROLES),
            "unit": "whole case dependency component",
            "random_row_split_allowed": False,
            "component_may_cross_roles": False,
            "justification_required": True,
            "production_role_assignment_authorized": False,
        },
        "scientific_scope": {
            "simulation_role": "SIMULATED_TARGET_CONDITIONAL_DEVELOPMENT",
            "independent_test_available": False,
            "mosaics_or_replicates_are_biological_replicates": False,
            "training_executed": False,
            "biological_performance_conclusion_allowed": False,
        },
    }


def validate_split(report, split):
    require(report.get("schema") == REPORT_SCHEMA, "unsupported dependency report")
    require(split.get("schema") == SPLIT_SCHEMA, "unsupported split schema")
    require(isinstance(split.get("justification"), str) and split["justification"].strip() and
            isinstance(split.get("decision_source"), str) and split["decision_source"].strip(),
            "explicit justification and decision_source are required")
    assignments = split.get("assignments")
    require(isinstance(assignments, list), "assignments must be a list")
    mapping = {}
    for item in assignments:
        require(isinstance(item, dict) and set(item) == {"case_id", "role"},
                "each assignment requires only case_id and role")
        require(item["case_id"] not in mapping and item["role"] in ROLES,
                "duplicate case or invalid FIT/SELECT/TEST role")
        mapping[item["case_id"]] = item["role"]
    expected = set(report["case_to_component"])
    require(set(mapping) == expected, "split must assign every and only the 12 authenticated cases")
    by_component = defaultdict(set)
    for case_id, role in mapping.items():
        by_component[report["case_to_component"][case_id]].add(role)
    leaked = {component: sorted(roles) for component, roles in by_component.items() if len(roles) > 1}
    require(not leaked, "dependency component crosses FIT/SELECT/TEST: " + json.dumps(leaked, sort_keys=True))
    return {
        "schema": "r02_lai_validated_case_split_v1",
        "status": "COHERENT_EXPLICIT_SPLIT_VALIDATED_NOT_TRAINING_ADMISSION",
        "dependency_report_sha256": split.get("dependency_report_sha256"),
        "justification": split["justification"],
        "decision_source": split["decision_source"],
        "assignments": [{"case_id": case_id, "role": mapping[case_id],
                         "component_id": report["case_to_component"][case_id]}
                        for case_id in sorted(mapping)],
        "roles_present": sorted(set(mapping.values())),
        "training_allowed": False,
        "independent_test_claimed": False,
    }


def emit_target_roles(report, validated, case_id, persons_path, output):
    require(validated.get("schema") == "r02_lai_validated_case_split_v1",
            "validated split required")
    assignments = {x["case_id"]: x for x in validated["assignments"]}
    require(case_id in assignments, "case absent from validated split")
    binding = assignments[case_id]
    require(binding["component_id"] == report["case_to_component"].get(case_id),
            "split component differs from report")
    require(binding["role"] in {"FIT", "SELECT"},
            "existing context builder supports FIT/SELECT only; TEST remains closed")
    with Path(persons_path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == ["sample_key_sha256", "donor_fold", "use"],
                "unexpected material person columns")
        people = [row["sample_key_sha256"] for row in reader]
    require(people and len(people) == len(set(people)), "empty or duplicate target identities")
    with Path(output).open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sample_key_sha256", "role", "component_id"))
        for person in people:
            writer.writerow((person, binding["role"], binding["component_id"]))
    return {"case_id": case_id, "role": binding["role"],
            "component_id": binding["component_id"], "people": len(people),
            "target_roles_sha256": sha256(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--requirements", required=True)
    build.add_argument("--design-manifest", required=True)
    build.add_argument("--scenarios", required=True)
    build.add_argument("--panel-roles", required=True)
    build.add_argument("--donor-usage")
    build.add_argument("--output", required=True)
    validate = sub.add_parser("validate-split")
    validate.add_argument("--dependency-report", required=True)
    validate.add_argument("--dependency-report-sha256", required=True)
    validate.add_argument("--split", required=True)
    validate.add_argument("--output", required=True)
    emit = sub.add_parser("emit-target-roles")
    emit.add_argument("--dependency-report", required=True)
    emit.add_argument("--dependency-report-sha256", required=True)
    emit.add_argument("--validated-split", required=True)
    emit.add_argument("--validated-split-sha256", required=True)
    emit.add_argument("--case-id", required=True)
    emit.add_argument("--persons", required=True)
    emit.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        value = build_report(args.requirements, args.design_manifest, args.scenarios,
                             args.panel_roles, args.donor_usage)
    elif args.command == "validate-split":
        require(sha256(args.dependency_report) == args.dependency_report_sha256,
                "dependency report SHA256 differs")
        report = read_json(args.dependency_report)
        split = read_json(args.split)
        require(split.get("dependency_report_sha256") == args.dependency_report_sha256,
                "split is not pinned to dependency report")
        value = validate_split(report, split)
    else:
        require(sha256(args.dependency_report) == args.dependency_report_sha256 and
                sha256(args.validated_split) == args.validated_split_sha256,
                "dependency or validated split SHA256 differs")
        value = emit_target_roles(read_json(args.dependency_report),
                                  read_json(args.validated_split), args.case_id,
                                  args.persons, args.output)
        print(json.dumps(value, sort_keys=True))
        return
    write_json(args.output, value)
    print(json.dumps({key: value[key] for key in ("schema", "status")}, sort_keys=True))


if __name__ == "__main__":
    main()
