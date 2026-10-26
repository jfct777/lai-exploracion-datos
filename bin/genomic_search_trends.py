#!/usr/bin/env python3
"""Describe authenticated development-search results without proposing or training.

The source is either an existing Optuna SQLite database opened in read-only mode
or an authenticated JSON export of the same public Optuna fields.  Every stored
complete receipt is checked again with genomic_adaptive_search.SearchController.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote
import shutil
import tempfile

import optuna
from optuna.trial import TrialState

import genomic_adaptive_search as search


EXPORT_SCHEMA = "genomic-adaptive-search-read-export/1"
REPORT_SCHEMA = "genomic-adaptive-search-trends/1"
WARNING = (
    "Las asociaciones entre hiperparámetros y pérdidas son descriptivas dentro "
    "del desarrollo registrado; no identifican efectos causales, un óptimo "
    "biológico ni justifican ampliar automáticamente un rango."
)
MEASURED_COST_METRICS = {
    "elapsed_seconds", "wall_seconds", "cpu_seconds", "cost_usd",
    "peak_memory_bytes", "peak_rss_bytes",
}


@dataclass(frozen=True)
class StoredTrial:
    number: int
    state: str
    params: dict[str, Any]
    value: float | None
    user_attrs: dict[str, Any]


@dataclass(frozen=True)
class StudySnapshot:
    user_attrs: dict[str, Any]
    trials: tuple[StoredTrial, ...]
    source: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path: Path, expected: str, label: str) -> str:
    search.require(path.is_file(), f"{label} does not exist")
    observed = sha256_file(path)
    search.require(search.valid_hash(expected) and observed == expected,
                   f"{label} SHA256 mismatch")
    return observed


def _trial_from_optuna(trial: optuna.trial.FrozenTrial) -> StoredTrial:
    value = None if trial.value is None else float(trial.value)
    return StoredTrial(trial.number, trial.state.name, dict(trial.params), value,
                       dict(trial.user_attrs))


def load_database(path: Path, expected_sha256: str, objective: str) -> StudySnapshot:
    """Read an authenticated self-contained copy; reject uncheckpointed WAL."""
    def no_wal():
        wal = Path(str(path) + "-wal")
        search.require(not wal.exists() or wal.stat().st_size == 0,
                       "nonempty SQLite WAL: export a consistent, self-contained snapshot first")
    no_wal()
    before = checked_hash(path, expected_sha256, "database")
    search.require(shutil.disk_usage(tempfile.gettempdir()).free > path.stat().st_size + 64 * 2**20,
                   "insufficient temporary disk for authenticated SQLite snapshot")
    with tempfile.TemporaryDirectory(prefix="r02-search-snapshot-") as folder:
        snapshot = Path(folder) / "search.sqlite3"
        shutil.copyfile(path, snapshot)
        checked_hash(snapshot, before, "copied database")
        no_wal()
        absolute = quote(str(snapshot.resolve()), safe="/")
        storage = f"sqlite:///file:{absolute}?mode=ro&uri=true"
        try:
            study = optuna.load_study(study_name=objective, storage=storage)
            trials = tuple(_trial_from_optuna(row) for row in study.get_trials(deepcopy=True))
            attributes = dict(study.user_attrs)
        except Exception as exc:
            raise search.ContractError(f"cannot read existing Optuna study: {exc}") from exc
        checked_hash(snapshot, before, "read database")
    after = sha256_file(path)
    no_wal()
    search.require(after == before, "database changed while the report was reading it")
    return StudySnapshot(attributes, trials,
                         {"kind": "optuna_sqlite_read_only", "sha256": before,
                          "consistency": "HASH_AUTHENTICATED_SELF_CONTAINED_COPY_NO_WAL"})


def load_export(path: Path, expected_sha256: str, objective: str) -> StudySnapshot:
    payload, observed = search.load_json(path, expected_sha256)
    search.keys(payload, ("schema", "objective", "study_user_attrs", "trials"),
                "study export")
    search.require(payload["schema"] == EXPORT_SCHEMA and payload["objective"] == objective,
                   "wrong export schema or objective")
    search.require(isinstance(payload["study_user_attrs"], dict)
                   and isinstance(payload["trials"], list), "invalid study export")
    trials = []
    for row in payload["trials"]:
        search.keys(row, ("number", "state", "params", "value", "user_attrs"),
                    "exported trial")
        search.require(search.integer(row["number"])
                       and row["state"] in {state.name for state in TrialState}
                       and isinstance(row["params"], dict)
                       and isinstance(row["user_attrs"], dict)
                       and (row["value"] is None or search.finite(row["value"])),
                       "invalid exported trial")
        trials.append(StoredTrial(row["number"], row["state"], row["params"],
                                  row["value"], row["user_attrs"]))
    search.require([row.number for row in trials] == list(range(len(trials))),
                   "exported trials must have contiguous original numbers")
    return StudySnapshot(dict(payload["study_user_attrs"]), tuple(trials),
                         {"kind": "authenticated_json_export", "sha256": observed})


def validate_snapshot(snapshot: StudySnapshot, plan: dict[str, Any], objective: str
                      ) -> list[dict[str, Any]]:
    """Reuse the controller contract without opening or mutating its state directory."""
    plan = search.validate_plan(plan)
    plan_sha256 = search.digest(plan)
    attrs = snapshot.user_attrs
    search.require(attrs.get("plan") == plan and attrs.get("plan_sha256") == plan_sha256,
                   "study plan differs from the authenticated plan")
    search.require(attrs.get("optuna_version") == search.OPTUNA_VERSION,
                   "study Optuna version differs from the controller contract")
    search.require(search.valid_hash(attrs.get("controller_sha256")),
                   "study lacks the controller code SHA256")
    receipts = attrs.get("receipts", {})
    failures = attrs.get("failures", {})
    search.require(isinstance(receipts, dict) and isinstance(failures, dict),
                   "invalid receipt or failure journal")

    # __new__ avoids SearchController.__init__, which creates a state directory.
    validator = search.SearchController.__new__(search.SearchController)
    validator.plan = plan
    validator.plan_sha256 = plan_sha256
    search.require(bool(snapshot.trials), "existing study contains no issued recipes")
    validated = []
    for trial in snapshot.trials:
        search.require(trial.state in {"COMPLETE", "FAIL", "RUNNING", "WAITING"},
                       "unsupported trial state under pruning=none")
        recipe = trial.user_attrs.get("recipe")
        search.require(isinstance(recipe, dict), "every issued trial needs its stored recipe")
        search.require(recipe.get("objective") == objective
                       and recipe.get("trial_number") == trial.number,
                       "stored recipe identity differs from its trial")
        search.require(recipe.get("recipe_sha256") == search.digest(
            {key: value for key, value in recipe.items() if key != "recipe_sha256"}),
            "stored recipe SHA256 mismatch")
        entry = receipts.get(str(trial.number))
        if trial.state == "COMPLETE":
            search.require(isinstance(entry, dict)
                           and set(entry) == {"sha256", "receipt", "summary"}
                           and search.valid_hash(entry["sha256"]),
                           "complete trial lacks an authenticated receipt journal entry")
            recomputed = validator.validate_receipt(entry["receipt"], recipe)
            search.require(search.canonical(recomputed) == search.canonical(entry["summary"]),
                           "stored receipt summary differs from contract recomputation")
            search.require(trial.value is not None
                           and math.isclose(trial.value, recomputed["selection_loss"],
                                            rel_tol=0.0, abs_tol=1e-15),
                           "Optuna value differs from authenticated absolute loss")
        else:
            search.require(entry is None, "unfinished/failed trial has a receipt journal entry")
        if trial.state == "FAIL":
            search.require(isinstance(failures.get(str(trial.number)), str),
                           "failed trial lacks its recorded reason")
        validated.append({"trial": trial, "recipe": recipe, "entry": entry})
    search.require(set(receipts) == {str(row["trial"].number) for row in validated
                                    if row["trial"].state == "COMPLETE"},
                   "receipt journal contains unknown trials")
    search.require(set(failures) == {str(row["trial"].number) for row in validated
                                    if row["trial"].state == "FAIL"},
                   "failure journal contains unknown trials")
    selection = attrs.get("frozen_selection")
    if selection is not None:
        complete = [row for row in validated if row["trial"].state == "COMPLETE"]
        eligible = complete
        if plan["schema"] == search.CONTROLLED_SCHEMA and "context_policy" not in plan:
            eligible = [row for row in complete
                        if row["entry"]["summary"]["eligible_development_selection"]]
        search.require(bool(eligible), "stored selection has no eligible complete recipe")
        best = min(eligible, key=lambda row: (row["trial"].value, row["trial"].number))
        search.require(selection.get("objective") == objective
                       and selection.get("plan_sha256") == plan_sha256
                       and selection.get("selected_trial") == best["trial"].number
                       and selection.get("selection_loss") == best["trial"].value
                       and selection.get("final_evaluation") == "NOT_ACCESSED"
                       and selection.get("training_launched") is False,
                       "stored selection differs from the original complete-recipe rule")
    return validated


def _json_key(value: Any) -> str:
    return search.canonical(value)


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _covered(values: list[Any]) -> dict[str, Any]:
    unique = list({_json_key(value): value for value in values}.values())
    numeric = all(type(value) in (int, float) for value in unique)
    unique.sort(key=(lambda value: value) if numeric else _json_key)
    return {
        "values": unique,
        "unique_count": len(unique),
        "minimum": min(unique) if unique and numeric else None,
        "maximum": max(unique) if unique and numeric else None,
    }


def _planned_domain(plan: dict[str, Any], parameter: str, observed: list[Any]) -> Any:
    if parameter in plan["search_space"]:
        value = plan["search_space"][parameter]
        if isinstance(value, dict) and set(value) >= {"low", "high"}:
            return {"low": value["low"], "high": value["high"],
                    "scale": "log" if value.get("log") else "linear"}
        return value
    return _covered(observed)["values"]


def _seed_stability(rows: list[dict[str, Any]], primary_metric: str) -> dict[str, Any]:
    ranges = []
    compatible_seed_counts = []
    for row in rows:
        if row["trial"].state != "COMPLETE":
            continue
        records = row["entry"]["receipt"]["records"]
        seeds: dict[int, list[float]] = {}
        signatures: dict[int, set[tuple[Any, ...]]] = {}
        for record in records:
            seeds.setdefault(record["seed"], []).append(record["metrics"][primary_metric])
            signatures.setdefault(record["seed"], set()).add(
                (record["fold"], record["arm"], record["evaluation_unit"], record["n_evaluated"]))
        if len(seeds) < 2 or len({frozenset(value) for value in signatures.values()}) != 1:
            continue
        values = [_mean(seed_values) for seed_values in seeds.values()]
        ranges.append(max(values) - min(values))
        compatible_seed_counts.append(len(seeds))
    if not ranges:
        return {"status": "EVIDENCIA_INSUFICIENTE",
                "reason": "menos de dos semillas con condiciones, pliegues, unidades y denominadores compatibles"}
    return {"status": "DESCRIPTIVO", "recipes": len(ranges),
            "minimum_compatible_seeds": min(compatible_seed_counts),
            "mean_within_recipe_loss_range": _mean(ranges),
            "maximum_within_recipe_loss_range": max(ranges)}


def build_report(plan: dict[str, Any], objective: str, snapshot: StudySnapshot,
                 validated: list[dict[str, Any]]) -> dict[str, Any]:
    spec = plan["objectives"][objective]
    primary = spec["primary_metric"]
    families = list(spec["families"])
    parameter_names = sorted({key for row in validated for key in row["recipe"]["config"]
                              if key != "family"})
    output_rows = []
    for family in families:
        family_rows = [row for row in validated
                       if row["recipe"]["config"].get("family") == family]
        states = {state: sum(row["trial"].state == state for row in family_rows)
                  for state in ("COMPLETE", "FAIL", "RUNNING", "WAITING")}
        incomplete = states["RUNNING"] + states["WAITING"]
        complete = [row for row in family_rows if row["trial"].state == "COMPLETE"]
        seed_stability = _seed_stability(complete, primary)
        for parameter in parameter_names:
            relevant = [row for row in family_rows if parameter in row["recipe"]["config"]]
            observed = [row["recipe"]["config"][parameter] for row in relevant]
            complete_parameter = [row for row in complete if parameter in row["recipe"]["config"]]
            groups: dict[str, dict[str, Any]] = {}
            for row in complete_parameter:
                value = row["recipe"]["config"][parameter]
                group = groups.setdefault(_json_key(value), {"value": value, "rows": []})
                group["rows"].append(row)
            summaries = []
            for group in groups.values():
                group_rows = group.pop("rows")
                measured = {}
                for metric in [primary, *spec["secondary_metrics"]]:
                    if metric in MEASURED_COST_METRICS:
                        measured[metric] = {
                            arm: _mean([row["entry"]["summary"]["arm_macro_losses"][arm][metric]
                                        for row in group_rows])
                            for arm in row["entry"]["summary"]["arm_macro_losses"]
                        }
                summaries.append({
                    "value": group["value"],
                    "complete_recipes": len(group_rows),
                    "mean_selection_loss": _mean([row["entry"]["summary"]["selection_loss"]
                                                   for row in group_rows]),
                    "minimum_selection_loss": min(row["entry"]["summary"]["selection_loss"]
                                                  for row in group_rows),
                    "mean_absolute_loss_by_condition": {
                        arm: _mean([row["entry"]["summary"]["arm_macro_losses"][arm][primary]
                                    for row in group_rows])
                        for arm in row["entry"]["summary"]["arm_macro_losses"]
                    },
                    "measured_cost_by_condition": measured,
                })
            summaries.sort(key=lambda item: _json_key(item["value"]))
            covered_complete = _covered([row["recipe"]["config"][parameter]
                                         for row in complete_parameter])
            evidence = "DESCRIPTIVO"
            reason = None
            boundary = None
            if len(complete_parameter) < 2:
                evidence, reason = "EVIDENCIA_INSUFICIENTE", "menos de dos recetas completas"
            elif covered_complete["unique_count"] < 2:
                evidence, reason = "EVIDENCIA_INSUFICIENTE", "el parámetro no varió"
            elif all(type(item["value"]) in (int, float) for item in summaries):
                best = min(summaries, key=lambda item: (item["mean_selection_loss"],
                                                        _json_key(item["value"])))
                ordered_values = [item["value"] for item in summaries]
                ordered_values.sort()
                at_lower = _json_key(best["value"]) == _json_key(ordered_values[0])
                at_upper = _json_key(best["value"]) == _json_key(ordered_values[-1])
                boundary = {
                    "best_observed_value": best["value"],
                    "at_lowest_covered_value": at_lower,
                    "at_highest_covered_value": at_upper,
                    "review_flag": at_lower or at_upper,
                    "meaning": "REVISAR_PROSPECTIVAMENTE_SIN_AMPLIACION_AUTOMATICA"
                               if at_lower or at_upper else "INTERIOR_DEL_RANGO_CUBIERTO",
                }
            cost_metrics = [name for name in [primary, *spec["secondary_metrics"]]
                            if name in MEASURED_COST_METRICS]
            output_rows.append({
                "family": family,
                "parameter": parameter,
                "planned_domain": _planned_domain(plan, parameter, observed),
                "covered_all_issued": _covered(observed),
                "covered_complete": covered_complete,
                "recipes": {"complete": states["COMPLETE"], "failed": states["FAIL"],
                            "incomplete": incomplete},
                "evidence_status": evidence,
                "insufficient_reason": reason,
                "loss_by_value": summaries,
                "best_value_at_covered_extreme": boundary,
                "seed_stability": seed_stability,
                "measured_cost_metrics": cost_metrics,
                "cost_status": "MEASURED_IN_AUTHENTICATED_RECEIPTS" if cost_metrics
                               else "NO_MEDIDO_EN_RECIBOS_AUTENTICADOS",
            })
    totals = {state: sum(row["trial"].state == state for row in validated)
              for state in ("COMPLETE", "FAIL", "RUNNING", "WAITING")}
    return {
        "schema": REPORT_SCHEMA,
        "objective": objective,
        "scope": "DEVELOPMENT_RESULTS_ONLY",
        "plan_sha256": search.digest(plan),
        "source": snapshot.source,
        "selection_rule": "mean_absolute_loss_across_complete_paired_conditions",
        "rare_effect_selection": plan.get("context_policy", {}).get("rare_effect_selection"),
        "recipe_totals": {"complete": totals["COMPLETE"], "failed": totals["FAIL"],
                          "incomplete": totals["RUNNING"] + totals["WAITING"]},
        "stored_selection": snapshot.user_attrs.get("frozen_selection"),
        "rows": output_rows,
        "warning": WARNING,
        "final_evaluation_accessed": False,
        "training_launched": False,
    }


def write_csv(path: Path, report: dict[str, Any]) -> None:
    fields = ("family", "parameter", "planned_domain", "covered_all_issued",
              "covered_complete", "recipes", "evidence_status", "insufficient_reason",
              "loss_by_value", "best_value_at_covered_extreme", "seed_stability",
              "measured_cost_metrics", "cost_status", "warning")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in report["rows"]:
            row = {key: source.get(key) for key in fields if key != "warning"}
            row["warning"] = report["warning"]
            for key, value in list(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = search.canonical(value)
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--objective", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--database", type=Path)
    source.add_argument("--study-export", type=Path)
    parser.add_argument("--database-sha256")
    parser.add_argument("--study-export-sha256")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, plan_file_sha256 = search.load_json(args.plan, args.plan_sha256)
    plan = search.validate_plan(plan)
    search.require(args.objective in plan["objectives"], "objective is absent from plan")
    if args.database is not None:
        search.require(args.database_sha256 is not None
                       and args.study_export_sha256 is None,
                       "database source requires only --database-sha256")
        snapshot = load_database(args.database, args.database_sha256, args.objective)
    else:
        search.require(args.study_export_sha256 is not None
                       and args.database_sha256 is None,
                       "export source requires only --study-export-sha256")
        snapshot = load_export(args.study_export, args.study_export_sha256, args.objective)
    validated = validate_snapshot(snapshot, plan, args.objective)
    report = build_report(plan, args.objective, snapshot, validated)
    report["plan_file_sha256"] = plan_file_sha256
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    write_csv(args.csv_out, report)
    print(search.canonical({"json": str(args.json_out), "csv": str(args.csv_out),
                            "rows": len(report["rows"]),
                            "recipe_totals": report["recipe_totals"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (search.ContractError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}")
