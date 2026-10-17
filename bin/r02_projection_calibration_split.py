#!/usr/bin/env python3
"""Build and verify K03 inner-FIT component splits without opening reserved roles.

This module validates mechanics only.  The current pair-complete calendar is
prospectively frozen and treats ancestry balance as a diagnostic, without an
ancestry minimum.  The legacy materialization interface still accepts explicit
ancestry-stratum requirements without defaults.  Every learned transformation,
graph and community must be rebuilt from the INNER_FIT rows in its plan.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import itertools
from pathlib import Path
from typing import Any

SCHEMA = "r02_k03_inner_fit_split_design_v1"
BUNDLE_SCHEMA = "r02_k03_inner_fit_split_bundle_v1"
PAIR_PLAN_SCHEMA = "r02_k03_inner_fit_pair_complete_plan_v1"
POLICIES = {
    "OBSERVED_MEAN_COMPLETE_REFERENCE_V1",
    "OBSERVED_MEAN_SINGLE_COMPONENT_DELETION_DIAGNOSTIC_V1",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest()


def read_tsv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == fields, f"{path.name} needs exact columns {fields}")
        rows = list(reader)
    require(rows and all(all(row[field].strip() == row[field] and row[field]
                             for field in fields) for row in rows),
            f"{path.name} contains empty or padded fields")
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve_frozen_sibling(declared_path: str, frozen_plan_path: Path) -> Path:
    """Resolve a frozen adjacent artifact after relocating the repository tree."""
    declared = Path(declared_path)
    if declared.is_file():
        return declared
    sibling = frozen_plan_path.parent / declared.name
    require(sibling.is_file(), f"Frozen artifact is unavailable: {declared.name}")
    return sibling


def roster_inventory(roster_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    people = read_tsv(roster_path, ["sample_id", "component", "role"])
    sample_ids = [row["sample_id"] for row in people]
    require(len(sample_ids) == len(set(sample_ids)), "Duplicate sample ID in frozen roster")
    require(all(row["role"] in {"FIT", "SELECT", "EVALUATE"} for row in people),
            "Unknown outer role in frozen roster")
    component_roles: dict[str, set[str]] = defaultdict(set)
    for row in people:
        component_roles[row["component"]].add(row["role"])
    require(all(len(roles) == 1 for roles in component_roles.values()),
            "A component crosses frozen FIT/SELECT/EVALUATE roles")
    by_role = Counter(row["role"] for row in people)
    fit_sizes = Counter(row["component"] for row in people if row["role"] == "FIT")
    require(len(fit_sizes) >= 2, "FIT needs at least two complete components")
    inventory = {
        "schema": "r02_k03_inner_fit_roster_inventory_v1",
        "status": "FIT_COMPONENTS_VERIFIED_SPLIT_DESIGN_REQUIRED",
        "roster": {"path": str(roster_path.resolve()), "sha256": sha256(roster_path)},
        "people_by_outer_role": dict(sorted(by_role.items())),
        "fit_people": sum(fit_sizes.values()),
        "fit_components": len(fit_sizes),
        "fit_component_size_distribution": {
            str(size): count for size, count in sorted(Counter(fit_sizes.values()).items())
        },
        "max_fit_component_size": max(fit_sizes.values()),
        "reserved_response_rows_read": False,
        "select_evaluate_genotypes_read": False,
        "scientific_parameters_supplied": False,
        "required_before_materialization": [
            "explicit fold identifiers and one fold assignment per FIT component",
            "an authenticated per-FIT-person ancestry-stratum file",
            "explicit required strata and minimum distinct-component counts for INNER_FIT and INNER_PROJECT",
            "scientific justification for the chosen split design",
        ],
        "biological_validation": False,
    }
    return people, inventory


def validate_design(design: dict[str, Any], assignment_path: Path,
                    diversity_path: Path, roster_digest: str) -> None:
    required = {
        "schema", "design_id", "roster_sha256", "assignment_sha256",
        "diversity_sha256", "fold_ids", "required_strata",
        "minimum_distinct_components_per_stratum", "diversity_semantics",
        "scientific_justification", "policies", "calibration_target",
    }
    require(set(design) == required and design["schema"] == SCHEMA,
            "Exact K03 inner-FIT design fields/schema required")
    require(design["roster_sha256"] == roster_digest,
            "Design is not bound to this frozen roster")
    require(design["assignment_sha256"] == sha256(assignment_path) and
            design["diversity_sha256"] == sha256(diversity_path),
            "Assignment/diversity bytes differ from the design")
    folds = design["fold_ids"]
    require(isinstance(folds, list) and len(folds) >= 2 and len(folds) == len(set(folds)) and
            all(isinstance(x, str) and x.strip() == x and x for x in folds),
            "At least two explicit unique fold identifiers required")
    strata = design["required_strata"]
    require(isinstance(strata, list) and strata and len(strata) == len(set(strata)) and
            all(isinstance(x, str) and x.strip() == x and x for x in strata),
            "Explicit unique ancestry strata required")
    minima = design["minimum_distinct_components_per_stratum"]
    require(set(minima) == {"INNER_FIT", "INNER_PROJECT"} and
            all(type(minima[k]) is int and minima[k] > 0 for k in minima),
            "Explicit positive component minima required for both internal roles")
    require(isinstance(design["diversity_semantics"], str) and
            len(design["diversity_semantics"].strip()) >= 20 and
            isinstance(design["scientific_justification"], str) and
            len(design["scientific_justification"].strip()) >= 40,
            "Diversity semantics and scientific split justification are required")
    require(design["policies"] == sorted(POLICIES),
            "The two predeclared projection policies must be compared together")
    require(design["calibration_target"] ==
            "GENETIC_PREDICTION_ERROR_GLOBAL_CONDITIONAL_AND_ASSIGNMENT_COVERAGE",
            "K03 calibration target changed")


def build_bundle(roster_path: Path, design_path: Path, assignment_path: Path,
                 diversity_path: Path, output: Path) -> dict[str, Any]:
    people, inventory = roster_inventory(roster_path)
    design = read_json(design_path)
    validate_design(design, assignment_path, diversity_path, inventory["roster"]["sha256"])
    assignments = read_tsv(assignment_path, ["component", "fold_id"])
    diversity = read_tsv(diversity_path, ["sample_id", "stratum"])

    fit_rows = [row for row in people if row["role"] == "FIT"]
    fit_components = {row["component"] for row in fit_rows}
    assigned = {row["component"]: row["fold_id"] for row in assignments}
    require(len(assigned) == len(assignments), "Duplicate component assignment")
    require(set(assigned) == fit_components, "Assignments must cover exactly the FIT components")
    require(set(assigned.values()) == set(design["fold_ids"]),
            "Assignment folds differ from the declared fold identifiers")

    diversity_by_person = {row["sample_id"]: row["stratum"] for row in diversity}
    require(len(diversity_by_person) == len(diversity), "Duplicate diversity sample ID")
    fit_ids = {row["sample_id"] for row in fit_rows}
    require(set(diversity_by_person) == fit_ids,
            "Diversity rows must cover exactly FIT; reserved roles are forbidden")
    require(set(diversity_by_person.values()) <= set(design["required_strata"]),
            "Undeclared ancestry stratum")
    require(set(diversity_by_person.values()) == set(design["required_strata"]),
            "Every required ancestry stratum must occur in FIT")
    require(not output.exists() and output.parent.is_dir(), "Output bundle must be new")
    output.mkdir(mode=0o700)

    files: dict[str, str] = {}
    fold_summaries = []
    index_by_sample = {row["sample_id"]: i for i, row in enumerate(people)}
    minima = design["minimum_distinct_components_per_stratum"]
    for fold in design["fold_ids"]:
        project_components = {component for component, value in assigned.items() if value == fold}
        inner_fit_components = fit_components - project_components
        require(project_components and inner_fit_components, f"Fold {fold} has an empty internal role")
        train = [row for row in fit_rows if row["component"] in inner_fit_components]
        project = [row for row in fit_rows if row["component"] in project_components]
        require({row["component"] for row in train}.isdisjoint(
                {row["component"] for row in project}), "Family-component leakage")
        counts: dict[str, dict[str, int]] = {}
        for role, rows in (("INNER_FIT", train), ("INNER_PROJECT", project)):
            by_stratum: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                by_stratum[diversity_by_person[row["sample_id"]]].add(row["component"])
            counts[role] = {stratum: len(by_stratum[stratum])
                            for stratum in design["required_strata"]}
            require(all(value >= minima[role] for value in counts[role].values()),
                    f"Fold {fold} fails declared ancestry-component support for {role}")
        plan = {
            "schema": "r02_k03_inner_fit_fold_plan_v1", "design_id": design["design_id"],
            "fold_id": fold, "roster_sha256": inventory["roster"]["sha256"],
            "inner_fit_indices": [index_by_sample[row["sample_id"]] for row in train],
            "inner_project_indices": [index_by_sample[row["sample_id"]] for row in project],
            "inner_fit_components": sorted(inner_fit_components),
            "inner_project_components": sorted(project_components),
            "distinct_components_by_stratum": counts,
            "learning_projection_order": [
                "RECOMPUTE_FREQUENCIES_AND_TRANSFORMATIONS_ON_INNER_FIT_ONLY",
                "BUILD_AFFINITY_GRAPH_WITH_INNER_FIT_ENDPOINTS_ONLY",
                "LEARN_COMMUNITIES_FROM_INNER_FIT_GRAPH_ONLY",
                "FIT_CONTINUOUS_AND_DISCRETE_COMPARATORS_ON_MATCHED_INNER_FIT_INPUTS",
                "PROJECT_INNER_PROJECT_QUERIES_USING_INNER_FIT_REFERENCES_ONLY",
                "AFTER_PREDICTIONS_ARE_FROZEN_SCORE_GENETIC_ERROR_AND_ASSIGNMENT_COVERAGE",
            ],
            "select_evaluate_access_allowed": False,
            "leiden_labels_used_as_truth": False,
        }
        name = f"fold-{fold}.plan.json"
        write_json(output / name, plan)
        files[name] = sha256(output / name)
        fold_summaries.append({
            "fold_id": fold, "inner_fit_people": len(train),
            "inner_project_people": len(project),
            "inner_fit_components": len(inner_fit_components),
            "inner_project_components": len(project_components),
            "plan": name,
        })
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "status": "SPLIT_MECHANICS_VERIFIED_SCIENTIFIC_EXECUTION_NOT_ADMITTED",
        "design": {"path": str(design_path.resolve()), "sha256": sha256(design_path),
                   "canonical_sha256": canonical_sha256(design)},
        "roster": inventory["roster"],
        "assignment_sha256": sha256(assignment_path),
        "diversity_sha256": sha256(diversity_path),
        "folds": fold_summaries,
        "files_sha256": files,
        "fit_people": inventory["fit_people"], "fit_components": inventory["fit_components"],
        "every_fit_component_projected_once": True,
        "family_component_leakage": False,
        "select_evaluate_access_allowed": False,
        "reserved_response_rows_read": False,
        "leiden_labels_used_as_truth": False,
        "calibration_executed": False,
        "biological_validation": False,
    }
    write_json(output / "manifest.json", manifest)
    verify_bundle(output, roster_path)
    return manifest


def verify_bundle(bundle: Path, roster_path: Path) -> dict[str, Any]:
    manifest = read_json(bundle / "manifest.json")
    require(manifest.get("schema") == BUNDLE_SCHEMA, "Unknown split bundle schema")
    require(manifest["roster"]["sha256"] == sha256(roster_path), "Roster identity differs")
    require(all((bundle / name).is_file() and sha256(bundle / name) == digest
                for name, digest in manifest["files_sha256"].items()),
            "Fold plan missing or altered")
    people, inventory = roster_inventory(roster_path)
    fit_indices = {i for i, row in enumerate(people) if row["role"] == "FIT"}
    projected: list[int] = []
    for summary in manifest["folds"]:
        plan = read_json(bundle / summary["plan"])
        train, project = set(plan["inner_fit_indices"]), set(plan["inner_project_indices"])
        require(train.isdisjoint(project) and train | project == fit_indices,
                "Fold indices leak or do not partition FIT")
        require(all(people[i]["role"] == "FIT" for i in train | project),
                "Reserved outer role entered an inner fold")
        train_components = {people[i]["component"] for i in train}
        project_components = {people[i]["component"] for i in project}
        require(train_components.isdisjoint(project_components), "Component crosses internal roles")
        projected.extend(project)
    require(len(projected) == len(set(projected)) == inventory["fit_people"] and
            set(projected) == fit_indices, "Every FIT person must be projected exactly once")
    return {"status": "PASS", "folds": len(manifest["folds"]),
            "fit_people": inventory["fit_people"], "fit_components": inventory["fit_components"]}


def validate_pair_complete_calendar(roster_path: Path, frozen_plan_path: Path) -> dict[str, Any]:
    """Validate the frozen five-group, all-pairs calendar without genomic data."""
    people, inventory = roster_inventory(roster_path)
    frozen = read_json(frozen_plan_path)
    required = {
        "status", "roster", "assignment", "deterministic_salt", "seed_search",
        "group_people", "group_components", "plans", "fit_people", "fit_components",
        "unique_between_component_pairs", "pair_prediction_contexts",
        "within_base_group_pairs", "query_contexts_per_person", "min_ancestry_components",
        "ancestry_strata_required", "ancestry_balance_status", "genotypes_read",
        "responses_read", "learning_executed", "source_code_sha256",
    }
    require(set(frozen) == required and
            frozen["status"] == "PROSPECTIVE_ASSIGNMENT_FROZEN_NOT_LEARNING_ADMISSION",
            "Exact prospective pair-complete frozen plan required")
    require(frozen["roster"]["sha256"] == inventory["roster"]["sha256"] == sha256(roster_path),
            "Frozen calendar roster identity differs")
    assignment_path = resolve_frozen_sibling(frozen["assignment"]["path"], frozen_plan_path)
    require(sha256(assignment_path) == frozen["assignment"]["sha256"],
            "Frozen component assignment missing or altered")
    assignment_rows = read_tsv(assignment_path, ["component", "fold_id"])
    assignment = {row["component"]: row["fold_id"] for row in assignment_rows}
    require(len(assignment) == len(assignment_rows), "Duplicate frozen component assignment")
    fit_indices = {i for i, row in enumerate(people) if row["role"] == "FIT"}
    fit_components = {people[i]["component"] for i in fit_indices}
    groups = [f"G{i}" for i in range(5)]
    require(set(assignment) == fit_components and set(assignment.values()) == set(groups),
            "Frozen assignment must cover exactly FIT in G0--G4")
    group_indices = {g: {i for i in fit_indices if assignment[people[i]["component"]] == g}
                     for g in groups}
    require([len(group_indices[g]) for g in groups] == frozen["group_people"] == [292] * 5,
            "Frozen group person counts differ")
    require([len({people[i]["component"] for i in group_indices[g]}) for g in groups] ==
            frozen["group_components"], "Frozen group component counts differ")
    expected_pairs = list(itertools.combinations(groups, 2))
    require(len(frozen["plans"]) == len(expected_pairs) == 10,
            "All ten unordered group-pair contexts are required")
    seen_contexts = set()
    project_count = Counter()
    context_sets = []
    for record in frozen["plans"]:
        require(set(record) == {"path", "sha256", "train", "project"},
                "Exact frozen context record required")
        path = resolve_frozen_sibling(record["path"], frozen_plan_path)
        require(sha256(path) == record["sha256"],
                "Frozen context plan missing or altered")
        plan = read_json(path)
        require(set(plan) == {"schema", "fold_id", "inner_fit_indices", "inner_project_indices",
                              "inner_fit_people", "inner_project_people", "target_pair_rule",
                              "pair_context_multiplicity", "score_weight_rule", "roster_sha256",
                              "select_evaluate_access_allowed", "leiden_labels_used_as_truth"}
                and plan["schema"] == PAIR_PLAN_SCHEMA,
                "Exact pair-complete context schema required")
        pair = tuple(plan["fold_id"].split("_"))
        require(len(pair) == 2 and pair in expected_pairs and pair not in seen_contexts,
                "Context ID is duplicate, reversed or outside G0--G4 pairs")
        seen_contexts.add(pair)
        project = set(plan["inner_project_indices"]); train = set(plan["inner_fit_indices"])
        expected_project = group_indices[pair[0]] | group_indices[pair[1]]
        require(project == expected_project and train == fit_indices - expected_project and
                train.isdisjoint(project), "Context indices differ from its two frozen groups")
        require(len(train) == record["train"] == plan["inner_fit_people"] == 876 and
                len(project) == record["project"] == plan["inner_project_people"] == 584,
                "Context must contain 876 INNER_FIT and 584 INNER_PROJECT people")
        require(plan["target_pair_rule"] == "BOTH_ENDPOINTS_IN_PROJECT_DIFFERENT_KINSHIP_COMPONENTS"
                and plan["pair_context_multiplicity"] == {"different_base_groups": 1,
                                                           "same_base_group": 4}
                and plan["score_weight_rule"] ==
                    "ORIGINAL_COMPONENT_PAIR_WEIGHT_DIVIDED_BY_CONTEXT_MULTIPLICITY"
                and plan["select_evaluate_access_allowed"] is False
                and plan["leiden_labels_used_as_truth"] is False,
                "Context scoring/access semantics changed")
        project_count.update(project); context_sets.append(project)
    require(seen_contexts == set(expected_pairs) and
            set(project_count) == fit_indices and set(project_count.values()) == {4} and
            frozen["query_contexts_per_person"] == 4,
            "Every FIT person must be projected in exactly four contexts")
    pair_contexts = Counter()
    eligible = 0
    ordered_fit = sorted(fit_indices)
    for x, i in enumerate(ordered_fit):
        for j in ordered_fit[x + 1:]:
            if people[i]["component"] == people[j]["component"]:
                continue
            eligible += 1
            pair_contexts[sum(i in context and j in context for context in context_sets)] += 1
    require(eligible == frozen["unique_between_component_pairs"] == 1_064_710 and
            set(pair_contexts) <= {1, 4} and
            sum(multiplicity * count for multiplicity, count in pair_contexts.items()) ==
                frozen["pair_prediction_contexts"] == 1_700_920,
            "Pair coverage/multiplicity differs from the frozen calendar")
    require(pair_contexts[4] == frozen["within_base_group_pairs"] == 212_070,
            "Same-base-group pair count differs from the frozen calendar")
    require(frozen["min_ancestry_components"] is None and
            frozen["ancestry_strata_required"] is False and
            frozen["ancestry_balance_status"] ==
                "NOT_MEASURED_DIAGNOSTIC_REQUIRED_NOT_SPLIT_BLOCKER",
            "Ancestry diagnostics cannot become an ungrounded split minimum")
    require(frozen["genotypes_read"] is False and frozen["responses_read"] is False and
            frozen["learning_executed"] is False,
            "Frozen metadata calendar cannot claim genomic execution")
    return {
        "schema": "r02_k03_pair_complete_calendar_admission_v1",
        "status": "PAIR_COMPLETE_METADATA_CALENDAR_VERIFIED_READY_FOR_STAGED_RECONSTRUCTION",
        "frozen_plan": {"path": str(frozen_plan_path.resolve()), "sha256": sha256(frozen_plan_path)},
        "roster": inventory["roster"], "assignment_sha256": sha256(assignment_path),
        "groups": 5, "contexts": 10, "inner_fit_people_per_context": 876,
        "inner_project_people_per_context": 584, "query_contexts_per_person": 4,
        "unique_between_component_pairs": eligible,
        "pair_prediction_contexts": frozen["pair_prediction_contexts"],
        "pair_context_counts_by_multiplicity": {str(k): v for k, v in sorted(pair_contexts.items())},
        "context_weight_by_multiplicity": {"1": 1.0, "4": 0.25},
        "ancestry_balance": "DIAGNOSTIC_NOT_SPLIT_BLOCKER",
        "genotypes_read": False, "responses_read": False, "learning_executed": False,
        "select_evaluate_access_allowed": False, "biological_validation": False,
        "next_computational_task": "G0_G1_R_FIRST_VIEW_STAGE_AUTHENTICATE_FILTER_SPECTRUM_ONE_DECLARED_SEARCH_EPISODE_AND_RESPONSE_PREPARATION",
    }


def exercise_inner_graph_path(roster_path: Path, fold_plan_path: Path,
                              affinity_path: Path, source_sha256: str,
                              max_memory_bytes: int,
                              producer_manifest_path: Path | None = None) -> dict[str, Any]:
    """Exercise the real spectrum/projection code on one declared inner split.

    A legacy table contains the complete outer-FIT eligible pair universe.  An
    authenticated INNER_FIT producer instead contains exactly train--train and
    project--train rows; this is the admitted route for calibration because it
    proves that frequencies, minor-allele orientation and affinities were
    rebuilt from raw genotypes.  Project--project rows never enter this path.
    """
    from r02_structure_continuous import FitRareGraphSpectrum

    people, inventory = roster_inventory(roster_path)
    plan = read_json(fold_plan_path)
    require(plan.get("schema") in {"r02_k03_inner_fit_fold_plan_v1", PAIR_PLAN_SCHEMA} and
            plan.get("roster_sha256") == inventory["roster"]["sha256"],
            "Fold plan is not bound to this roster")
    train = set(plan["inner_fit_indices"])
    project = set(plan["inner_project_indices"])
    outer_fit = {i for i, row in enumerate(people) if row["role"] == "FIT"}
    require(train and project and train.isdisjoint(project) and train | project == outer_fit,
            "Full outer FIT cannot be reused as inner learning")
    rows = read_tsv(affinity_path, ["left_index", "right_index", "observed", "affinity",
                                            "joint_callable_sites", "value_denominator_sites",
                                            "candidate_sites"])
    producer = None
    if producer_manifest_path is not None:
        producer_manifest_path = Path(producer_manifest_path)
        producer = read_json(producer_manifest_path)
        import r02_k03_inner_fit_affinities as inner_producer
        raw = producer.get("raw_genotype_manifest", {})
        raw_name = raw.get("path") if isinstance(raw, dict) else None
        raw_path = (producer_manifest_path.parent / raw_name
                    if isinstance(raw_name, str) else producer_manifest_path.parent / "__invalid__")
        transform_path = producer_manifest_path.parent / "inner_fit_transformations.private.bin"
        require(producer.get("schema") == "r02_k03_inner_fit_rare_affinities_v1"
                and producer.get("status") ==
                    "INNER_FIT_FREQUENCIES_ORIENTATION_TRANSFORMATION_AND_AFFINITIES_COMPLETE"
                and producer.get("roster_sha256") == sha256(roster_path)
                and producer.get("fold_plan_sha256") == sha256(fold_plan_path)
                and producer.get("learning_indices") == sorted(train)
                and producer.get("projection_indices") == sorted(project)
                and producer.get("frequency_scope") == "INNER_FIT_ONLY"
                and producer.get("minor_allele_scope") == "INNER_FIT_ONLY"
                and producer.get("transformation_scope") == "INNER_FIT_ONLY"
                and producer.get("candidate_ascertainment") == inner_producer.STRICT_INNER_FIT_ASCERTAINMENT
                and producer.get("genotype_encoding") == inner_producer.GT_CODE_ENCODING
                and producer.get("outer_fit_learned_inputs_used") is False
                and producer.get("select_evaluate_genotypes_read") is False
                and source_sha256 == sha256(producer_manifest_path)
                and producer.get("implementation_sha256") == inner_producer.implementation_sha256()
                and set(raw) == {"path", "sha256"}
                and raw_name == "raw_genotype_manifest.json" and raw_path.is_file()
                and sha256(raw_path) == raw["sha256"]
                and producer.get("files_sha256", {}).get(raw_path.name) == raw["sha256"]
                and producer.get("files_sha256", {}).get(affinity_path.name) == sha256(affinity_path),
                "Authenticated raw-genotype INNER_FIT producer required")
        require(transform_path.is_file()
                and producer["files_sha256"].get(transform_path.name) == sha256(transform_path),
                "INNER_FIT transformation evidence is absent or changed")
        train_sizes = Counter(people[i]["component"] for i in train)
        expected = (len(train) * (len(train) - 1) // 2
                    - sum(n * (n - 1) // 2 for n in train_sizes.values())
                    + len(project) * len(train))
    else:
        expected = len(outer_fit) * (len(outer_fit) - 1) // 2
        component_sizes = Counter(people[i]["component"] for i in outer_fit)
        expected -= sum(n * (n - 1) // 2 for n in component_sizes.values())
    require(len(rows) == expected,
            ("Authenticated INNER_FIT affinity input does not contain its exact eligible pair universe"
             if producer is not None else
             "Affinity input is not the complete outer-FIT eligible universe"))
    seen = set()
    train_edges = []
    projected: dict[int, dict[int, float]] = {i: {} for i in project}
    for row in rows:
        i, j = int(row["left_index"]), int(row["right_index"])
        require(i < j and i in outer_fit and j in outer_fit and (i, j) not in seen,
                "Affinity key is duplicate, reversed or outside outer FIT")
        seen.add((i, j))
        require(people[i]["component"] != people[j]["component"] or
                producer is not None and ((i in project) != (j in project)),
                "Within-component INNER_FIT graph row forbidden")
        joint, denominator, candidate = (int(row[key]) for key in
            ("joint_callable_sites", "value_denominator_sites", "candidate_sites"))
        require(0 <= denominator <= joint <= candidate and candidate > 0,
                "Invalid genomic support denominator")
        observed = row["observed"] == "1"
        require(observed == (denominator > 0)
                and (observed or row["affinity"] == "NA"),
                "Unknown affinity must remain NA; observed affinity needs positive support")
        value = float(row["affinity"]) if observed else None
        if i in train and j in train:
            require(observed, "INNER_FIT graph needs complete observed pair support")
            train_edges.append((i, j, value))
        elif i in project and j in train:
            if observed:
                projected[i][j] = value
        elif j in project and i in train:
            if observed:
                projected[j][i] = value
        elif producer is not None:
            require(False, "Authenticated producer included a project--project row")
    semantics = (dict(family="R", measure="INNER_FIT_MINOR_CARRIER_JACCARD",
                      units="DIMENSIONLESS_INTERSECTION_OVER_UNION",
                      minimum=0.0, maximum=1.0)
                 if producer is not None else None)
    graph = FitRareGraphSpectrum(
        people, train_edges, source_sha256=source_sha256,
        max_memory_bytes=max_memory_bytes,
        affinity_semantics=semantics,
        inner_fit_indices=sorted(train), inner_project_indices=sorted(project),
    )
    outputs = [graph.project(i, projected[i], source_sha256=source_sha256)
               for i in sorted(project)]
    if producer is None:
        require(all(row["observed_fit_count"] == len(train) for row in outputs),
                "Projection references do not equal exact inner FIT")
    return {
        "status": "INNER_FIT_GRAPH_AND_EXACT_REFERENCE_PROJECTION_EXECUTED",
        "fold_id": plan["fold_id"], "inner_fit_people": len(train),
        "inner_project_people": len(project), "fit_graph_sha256": graph.graph_sha256,
        "affinity_sha256": sha256(affinity_path),
        "projection_status_counts": dict(Counter(row["status"] for row in outputs)),
        "exact_inner_fit_reference_count": len(train),
        "inner_fit_producer_authenticated": producer is not None,
        "producer_manifest_sha256": (sha256(producer_manifest_path)
                                      if producer_manifest_path is not None else None),
        "full_outer_fit_used_for_learning": False,
        "project_project_edges_used": False,
        "select_evaluate_access_allowed": False,
        "target_read": False, "biological_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inventory")
    inspect.add_argument("--roster", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("materialize")
    for name in ("roster", "design", "assignments", "diversity", "output"):
        build.add_argument("--" + name, type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--roster", type=Path, required=True)
    exercise = sub.add_parser("exercise-path")
    exercise.add_argument("--roster", type=Path, required=True)
    exercise.add_argument("--fold-plan", type=Path, required=True)
    exercise.add_argument("--affinities", type=Path, required=True)
    exercise.add_argument("--source-sha256", required=True)
    exercise.add_argument("--max-memory-bytes", type=int, required=True)
    exercise.add_argument("--producer-manifest", type=Path)
    exercise.add_argument("--output", type=Path, required=True)
    calendar = sub.add_parser("admit-calendar")
    calendar.add_argument("--roster", type=Path, required=True)
    calendar.add_argument("--frozen-plan", type=Path, required=True)
    calendar.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        _, value = roster_inventory(args.roster)
        require(not args.output.exists() and args.output.parent.is_dir(), "Inventory output must be new")
        write_json(args.output, value)
    elif args.command == "materialize":
        value = build_bundle(args.roster, args.design, args.assignments, args.diversity, args.output)
    elif args.command == "verify":
        value = verify_bundle(args.bundle, args.roster)
    elif args.command == "exercise-path":
        value = exercise_inner_graph_path(args.roster, args.fold_plan, args.affinities,
                                          args.source_sha256, args.max_memory_bytes,
                                          args.producer_manifest)
        require(not args.output.exists() and args.output.parent.is_dir(), "Result output must be new")
        write_json(args.output, value)
    else:
        value = validate_pair_complete_calendar(args.roster, args.frozen_plan)
        require(not args.output.exists() and args.output.parent.is_dir(), "Admission output must be new")
        write_json(args.output, value)
    print(json.dumps({"status": value["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
