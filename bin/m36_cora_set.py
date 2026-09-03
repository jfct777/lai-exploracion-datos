#!/usr/bin/env python3
"""Validate and plan the synthetic-smoke surface of M36 CORA-Set.

M36 treats rare variants as individual event tokens.  It does not use M14 or
M16.5 labels and does not fit a model. Explicit real training is implemented
only in ``m36_cora_train.py`` after a materialization receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Keep direct unit-test loading and staged Nextflow execution on the same
# import path without relying on the caller's current directory.
BIN_DIR = Path(__file__).resolve().parent
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))
from m36_cora_models import available_specs


EVENT_CLASSES = ("AC2_HET", "AC2_HOMALT", "MAC3_10")
TARGET_SOURCES = (
    "common_wgs_ibd", "chromopainter_common", "asibd_refined_ibd_gnomix_stratified_exploratory",
)
REQUIRED_EVENT_COLUMNS = {
    "sample_id", "event_id", "chrom", "position", "mac", "genotype",
    "callability", "mutation_context", "cm", "common_copying_context",
}
REAL_EVENT_STATE_COLUMNS = {"genotype_state", "evaluable_mask"}
GENOTYPE_STATES = {"ALT_CARRIER", "ZERO_EVALUABLE", "MISSING"}
REQUIRED_COVARIATE_COLUMNS = {
    "sample_id", "rare_burden", "rare_callability", "cohort", "Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS",
}
REQUIRED_COMPONENT_COLUMNS = {"sample_id", "pcrelate_component"}
REQUIRED_TARGET_COLUMNS = {
    "sample_i", "sample_j", "target_chrom", "target_source", "target",
}


class ContractError(ValueError):
    """Raised for any M36 input or protocol violation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--covariates", required=True, type=Path)
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--feature-chrom", required=True)
    parser.add_argument("--model-families", required=True)
    parser.add_argument("--halving-budgets", required=True)
    parser.add_argument("--halving-eta", required=True, type=int)
    parser.add_argument("--n-folds", required=True, type=int)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ContractError(f"Missing input: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    if not rows:
        raise ContractError(f"{label} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ContractError(f"{label} missing columns: {sorted(missing)}")


def as_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ContractError(f"{label} must be an integer") from error


def as_float(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ContractError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ContractError(f"{label} must be finite")
    return number


def classify_event(rows: list[dict[str, str]]) -> str:
    """Classify an event from dosages; AC2 hom-alt is not a doubleton pair."""
    mac = {as_int(row["mac"], "mac") for row in rows}
    if len(mac) != 1:
        raise ContractError(f"event {rows[0]['event_id']} has inconsistent MAC")
    dosages = [as_int(row["genotype"], "genotype") for row in rows]
    if any(value not in (1, 2) for value in dosages):
        raise ContractError(f"event {rows[0]['event_id']} has non-carrier dosage")
    if sum(dosages) != next(iter(mac)):
        raise ContractError(f"event {rows[0]['event_id']} dosage does not equal MAC")
    if mac == {2} and sorted(dosages) == [1, 1]:
        return "AC2_HET"
    if mac == {2} and dosages == [2]:
        return "AC2_HOMALT"
    if len(mac) == 1 and 3 <= next(iter(mac)) <= 10:
        return "MAC3_10"
    raise ContractError(f"event {rows[0]['event_id']} is outside AC2/MAC3-10 scope")


def validate_contract(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "stage": "M36_CORA_SET_EXPLORATORY",
        "execution_scope": "synthetic_smoke_or_explicit_real_external_target_training; no_LAI_or_biological_superiority_claim",
        "event_classes": list(EVENT_CLASSES),
        "target_sources": list(TARGET_SOURCES),
        "m14_as_truth": False,
        "cross_chromosome_required": True,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ContractError(f"M36 contract drift at {key}")
    return data


def validate_inputs(
    events: list[dict[str, str]], covariates: list[dict[str, str]],
    components: list[dict[str, str]], targets: list[dict[str, str]], feature_chrom: str,
) -> tuple[dict[str, str], dict[str, str], Counter[str]]:
    require_columns(events, REQUIRED_EVENT_COLUMNS, "events")
    require_columns(covariates, REQUIRED_COVARIATE_COLUMNS, "covariates")
    require_columns(components, REQUIRED_COMPONENT_COLUMNS, "PC-Relate components")
    require_columns(targets, REQUIRED_TARGET_COLUMNS, "external targets")
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_carriers: set[tuple[str, str]] = set()
    for row in events:
        if row["chrom"] != feature_chrom:
            raise ContractError("events must be restricted to the declared feature chromosome")
        key = (row["event_id"], row["sample_id"])
        if key in seen_carriers:
            raise ContractError("duplicate sample/event carrier row")
        seen_carriers.add(key)
        if as_int(row["position"], "position") <= 0 or as_float(row["cm"], "cm") < 0:
            raise ContractError("event coordinates must be positive physical and nonnegative cM")
        callability = as_float(row["callability"], "callability")
        if not 0 <= callability <= 1:
            raise ContractError("event callability must lie in [0, 1]")
        state = row.get("genotype_state", "ALT_CARRIER")
        if state not in GENOTYPE_STATES:
            raise ContractError("genotype_state must be ALT_CARRIER, ZERO_EVALUABLE or MISSING")
        if "evaluable_mask" in row:
            evaluable = as_int(row["evaluable_mask"], "evaluable_mask")
            if evaluable not in (0, 1):
                raise ContractError("evaluable_mask must be zero or one")
            if (state == "MISSING") != (evaluable == 0):
                raise ContractError("MISSING must have evaluable_mask=0; observed genotypes require one")
        if state == "ALT_CARRIER":
            by_event[row["event_id"]].append(row)
        elif state == "ZERO_EVALUABLE" and as_int(row["genotype"], "genotype") != 0:
            raise ContractError("ZERO_EVALUABLE must have genotype zero")
    # A production chromosome can legitimately lack a rare stratum; absence is
    # reported rather than converted into a schema failure.  The fixture tests
    # exercise all three classes explicitly.
    classes = Counter(classify_event(rows) for rows in by_event.values())

    covariate_map = {row["sample_id"]: row for row in covariates}
    component_map = {row["sample_id"]: row["pcrelate_component"] for row in components}
    if len(covariate_map) != len(covariates) or len(component_map) != len(components):
        raise ContractError("covariates and components require unique sample_id")
    observed_samples = {row["sample_id"] for row in events}
    for row in targets:
        if row["target_source"] not in TARGET_SOURCES:
            raise ContractError("target is not permitted external common-WGS IBD, Gnomix-stratified asIBD, or ChromoPainter")
        if row["target_chrom"] == feature_chrom:
            raise ContractError("target chromosome must differ from feature chromosome")
        if row["sample_i"] == row["sample_j"]:
            raise ContractError("external target pair cannot self-pair")
        if as_float(row["target"], "target") < 0:
            raise ContractError("external target must be nonnegative")
        observed_samples.update((row["sample_i"], row["sample_j"]))
    missing_covariates = observed_samples - set(covariate_map)
    missing_components = observed_samples - set(component_map)
    if missing_covariates or missing_components:
        raise ContractError(
            "sample lacks covariate or PC-Relate component: "
            f"covariates={sorted(missing_covariates)}, components={sorted(missing_components)}"
        )
    for row in covariates:
        if as_float(row["rare_burden"], "rare_burden") < 0:
            raise ContractError("rare_burden must be nonnegative")
        if not 0 <= as_float(row["rare_callability"], "rare_callability") <= 1:
            raise ContractError("rare_callability must lie in [0, 1]")
        q_sum = sum(as_float(row[column], column) for column in ("Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS"))
        if not 0.99 <= q_sum <= 1.01:
            raise ContractError("Q_AFR/Q_EUR/Q_NAM/Q_EAS must sum approximately to one")
    return covariate_map, component_map, classes


def component_folds(component_map: dict[str, str], n_folds: int) -> dict[str, int]:
    if n_folds < 2:
        raise ContractError("M36 requires at least two outer folds")
    sizes = Counter(component_map.values())
    loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for component, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(n_folds), key=lambda index: (loads[index], index))
        assignment[component] = fold
        loads[fold] += size
    return assignment


def pair_partition(targets: list[dict[str, str]], component_map: dict[str, str],
                   assignment: dict[str, int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in targets:
        left = assignment[component_map[row["sample_i"]]]
        right = assignment[component_map[row["sample_j"]]]
        counts["assessment" if left == right else "cross_fold_excluded"] += 1
    if not counts["assessment"]:
        raise ContractError("component-disjoint folds leave no assessable external pairs")
    return counts


def successive_halving(families: tuple[str, ...], budgets: tuple[int, ...], eta: int) -> list[dict[str, Any]]:
    if eta < 2 or len(budgets) < 2 or any(left >= right for left, right in zip(budgets, budgets[1:])):
        raise ContractError("successive-halving budgets must be strictly increasing with eta >= 2")
    candidates = available_specs(families)
    plan: list[dict[str, Any]] = []
    for stage, budget in enumerate(budgets):
        for rank, spec in enumerate(candidates):
            plan.append({
                "stage": stage,
                "budget": budget,
                "candidate_rank": rank,
                "family": spec.family,
                "hidden_dim": spec.hidden_dim,
                "depth": spec.depth,
                "attention_heads": spec.attention_heads,
                "inducing_points": spec.inducing_points,
            })
        if stage < len(budgets) - 1:
            candidates = candidates[: max(1, math.ceil(len(candidates) / eta))]
    return plan


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["stage"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def event_tokens(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Materialise individual event-set tokens without fitting vocabularies.

    ``mutation_context`` remains categorical so a future authorised learner must
    fit its embedding vocabulary inside FIT, rather than leaking SCORE contexts.
    """
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in events:
        by_event[row["event_id"]].append(row)
    tokens: list[dict[str, Any]] = []
    for event_id, rows in sorted(by_event.items()):
        carrier_rows = [row for row in rows if row.get("genotype_state", "ALT_CARRIER") == "ALT_CARRIER"]
        event_class = classify_event(carrier_rows)
        for row in sorted(carrier_rows, key=lambda item: item["sample_id"]):
            tokens.append({
                "sample_id": row["sample_id"],
                "event_id": event_id,
                "event_class": event_class,
                "genotype_dosage": as_int(row["genotype"], "genotype") / 2.0,
                "mac_scaled": as_int(row["mac"], "mac") / 10.0,
                "callability": as_float(row["callability"], "callability"),
                "cm": as_float(row["cm"], "cm"),
                "common_copying_context": as_float(
                    row["common_copying_context"], "common_copying_context"
                ),
                "common_copying_context_available": as_int(row.get("common_copying_context_available", "1"), "common_copying_context_available"),
                "mutation_context_available": as_int(row.get("mutation_context_available", "1"), "mutation_context_available"),
                "is_ac2_het": int(event_class == "AC2_HET"),
                "is_ac2_homalt": int(event_class == "AC2_HOMALT"),
                "is_mac3_10": int(event_class == "MAC3_10"),
                "mutation_context": row["mutation_context"],
            })
    return tokens


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.smoke_only:
        raise ContractError("this planner requires --smoke-only; real training uses the receipt-bound trainer")
    contract = validate_contract(args.contract)
    events, covariates, components, targets = (
        read_tsv(args.events), read_tsv(args.covariates), read_tsv(args.components), read_tsv(args.targets)
    )
    _, component_map, classes = validate_inputs(events, covariates, components, targets, args.feature_chrom)
    families = tuple(value.strip() for value in args.model_families.split(",") if value.strip())
    budgets = tuple(as_int(value, "halving budget") for value in args.halving_budgets.split(","))
    assignment = component_folds(component_map, args.n_folds)
    partitions = pair_partition(targets, component_map, assignment)
    plan = successive_halving(families, budgets, args.halving_eta)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.outdir / "m36_cora_event_tokens.tsv", event_tokens(events))
    write_tsv(args.outdir / "m36_cora_trial_plan.tsv", plan)
    fold_rows = [
        {"pcrelate_component": component, "outer_fold": fold,
         "n_individuals": sum(value == component for value in component_map.values())}
        for component, fold in sorted(assignment.items())
    ]
    write_tsv(args.outdir / "m36_cora_component_folds.tsv", fold_rows)
    summary = {
        "stage": contract["stage"],
        "smoke_only": True,
        "training_executed": False,
        "m14_as_truth": False,
        "feature_chrom": args.feature_chrom,
        "target_chromosomes": sorted({row["target_chrom"] for row in targets}),
        "target_sources": sorted({row["target_source"] for row in targets}),
        "event_class_counts": dict(sorted(classes.items())),
        "n_individuals": len(component_map),
        "n_pcrelate_components": len(set(component_map.values())),
        "pair_partitions": dict(sorted(partitions.items())),
        "controls": ["rare_burden", "rare_callability", "cohort", "Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS"],
        "mutation_context_encoding": "categorical_vocabulary_fit_only_future_training",
        "successive_halving_candidates": len(plan),
    }
    (args.outdir / "m36_cora_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    try:
        run(parse_args())
    except ContractError as error:
        raise SystemExit(f"M36 contract error: {error}") from error


if __name__ == "__main__":
    main()
