#!/usr/bin/env python3
"""Local, development-only Optuna ask/tell controller; never launches training.

One trial is one complete paired recipe, not one fold, seed or arm. SQLite
stores the immutable plan, proposals and metric receipts. A POSIX file lock
serializes this controller on one local filesystem; this is not a distributed
scheduler. See docs/genomic_adaptive_search.md for the adapter contract.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import warnings

import optuna
from optuna.samplers import PartialFixedSampler, TPESampler
from optuna.trial import TrialState


SCHEMA = "genomic-adaptive-search/1"
CONTEXTUAL_SCHEMA = "genomic-adaptive-search/2"
CONTEXTUAL_FAMILIES = {
    "structure": ("structure_deep_sets", "structure_local_attention"),
    "lai": ("lai_cnn", "lai_local_attention"),
}
CONTEXTUAL_FIELDS = {
    "common_radius_cm", "rare_radius_cm", "width", "depth", "dropout", "heads", "learning_rate",
}
OPTUNA_VERSION = "4.9.0"
HARD_MAX_RECIPES = 24
ARMS = ("NONE", "RARE")
HASH = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")


class ContractError(ValueError):
    """An input or persisted record violates the frozen search contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            f"{label}: unexpected or missing fields")


def load_json(path, expected_sha256=None):
    """Authenticate exact receipt bytes; reject ambiguous JSON duplicate keys."""
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    with Path(path).open("rb") as handle:
        raw = handle.read(16 * 1024 * 1024 + 1)
    require(len(raw) <= 16 * 1024 * 1024, "plan/receipt exceeds the 16 MiB metadata limit")
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        require(valid_hash(expected_sha256) and observed == expected_sha256,
                "file SHA256 mismatch")
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite JSON")))
    canonical(value)
    return value, observed


def validate_plan(plan):
    keys(plan, ("schema", "campaign_id", "scope", "max_recipes", "startup_trials",
                "sampler_seed", "training_seeds", "search_space", "objectives",
                "provenance", "pruning"), "plan")
    require(plan["schema"] in (SCHEMA, CONTEXTUAL_SCHEMA) and plan["scope"] == "DEVELOPMENT",
            "supported versioned development plan required")
    contextual = plan["schema"] == CONTEXTUAL_SCHEMA
    require(isinstance(plan["campaign_id"], str) and IDENTIFIER.fullmatch(plan["campaign_id"]),
            "invalid campaign_id")
    require(integer(plan["max_recipes"], 1) and plan["max_recipes"] <= HARD_MAX_RECIPES,
            "maximum is 24 recipes per objective, including failures")
    require(integer(plan["startup_trials"], 1) and plan["startup_trials"] <= plan["max_recipes"],
            "invalid startup budget")
    require(plan["pruning"] == "none", "only paired full-budget evaluation is supported; pruning must be none")
    require(integer(plan["sampler_seed"]), "invalid sampler seed")
    seeds = plan["training_seeds"]
    require(isinstance(seeds, list) and seeds and all(integer(s) for s in seeds)
            and len(set(seeds)) == len(seeds), "training seeds must be fixed unique integers")
    space = plan["search_space"]
    radius_name = "common_radius_cm" if contextual else "radius_cm"
    keys(space, CONTEXTUAL_FIELDS if contextual else ("radius_cm", "width", "learning_rate"), "search_space")
    checks = {radius_name: lambda x: finite(x) and x > 0, "width": lambda x: integer(x, 1)}
    if contextual:
        checks.update({"rare_radius_cm": lambda x: finite(x) and x > 0,
                       "depth": lambda x: integer(x, 1), "heads": lambda x: integer(x, 1),
                       "dropout": lambda x: finite(x) and x < 1})
    for name, check in checks.items():
        values = space[name]
        require(isinstance(values, list) and values and all(check(x) for x in values),
                f"invalid {name} choices")
        require(len(set(values)) == len(values), f"invalid {name} choices")
    if contextual:
        # A fixed categorical domain avoids invalid trials and Optuna's
        # forbidden dynamic categorical distributions after choosing width.
        require(all(width % heads == 0 for width in space["width"] for heads in space["heads"]),
                "every width must be divisible by every heads choice")
    keys(space["learning_rate"], ("low", "high", "log"), "learning_rate")
    low, high = space["learning_rate"]["low"], space["learning_rate"]["high"]
    require(finite(low) and finite(high) and 0 < low < high and space["learning_rate"]["log"] is True,
            "learning rate requires a positive log-uniform interval")
    require(plan["startup_trials"] == 2 * len(space[radius_name]) * len(space["width"]),
            "startup budget must cover the two-family/radius/width product exactly")
    provenance = plan["provenance"]
    keys(provenance, ("dataset_sha256", "split_manifest_sha256", "feature_contract_sha256",
                      "trainer_code_sha256", "container_digest", "development_only_audit_sha256"), "provenance")
    require(all(valid_hash(value) for key, value in provenance.items() if key != "container_digest"),
            "all provenance digests are required")
    require(isinstance(provenance["container_digest"], str)
            and provenance["container_digest"].startswith("sha256:")
            and valid_hash(provenance["container_digest"][7:]), "container must be pinned by digest")
    require(isinstance(plan["objectives"], dict) and bool(plan["objectives"])
            and set(plan["objectives"]) <= {"structure", "lai"},
            "freeze a nonempty subset of structure/lai objectives")
    for name, spec in plan["objectives"].items():
        keys(spec, ("families", "primary_metric", "secondary_metrics", "folds", "fixed_config",
                    "checkpoint_step", "arm_input_sha256", "range_rationale", "evaluation_unit"), name)
        require(isinstance(spec["families"], list) and len(spec["families"]) == 2
                and len(set(spec["families"])) == 2
                and all(isinstance(f, str) and IDENTIFIER.fullmatch(f) for f in spec["families"]),
                "exactly two named families per objective required")
        if contextual:
            require(set(spec["families"]) == set(CONTEXTUAL_FAMILIES[name]),
                    f"v2 {name} requires its canonical contextual families")
        require(isinstance(spec["secondary_metrics"], list), "secondary_metrics must be a list")
        metrics = [spec["primary_metric"], *spec["secondary_metrics"]]
        require(len(set(metrics)) == len(metrics)
                and all(isinstance(m, str) and IDENTIFIER.fullmatch(m) for m in metrics), "invalid metric names")
        require(isinstance(spec["fixed_config"], dict) and spec["fixed_config"], "fixed trainer config is required")
        reserved = {"family", "radius_cm", "width", "learning_rate", "seed", "arm", "fold"}
        if contextual:
            reserved |= CONTEXTUAL_FIELDS
        require(not set(spec["fixed_config"]) & reserved,
                "fixed config must not override sampled or paired fields")
        require(integer(spec["checkpoint_step"], 1), "positive fixed checkpoint step required")
        require(spec["evaluation_unit"] in ("person_macro", "family_macro", "donor_macro", "component_macro"),
                "freeze a dependence-aware evaluation unit per objective")
        require(isinstance(spec["range_rationale"], str) and len(spec["range_rationale"].strip()) >= 20,
                "record the dataset-specific rationale; ranges are not established optima")
        keys(spec["arm_input_sha256"], ARMS, "arm input manifests")
        require(all(valid_hash(v) for v in spec["arm_input_sha256"].values()), "arm input hashes required")
        require(spec["arm_input_sha256"]["NONE"] != spec["arm_input_sha256"]["RARE"], "distinct arm manifests required")
        folds = spec["folds"]
        require(isinstance(folds, list) and len(folds) >= 2, "at least two development folds required")
        ids = []
        for fold in folds:
            keys(fold, ("id", "fit_groups_sha256", "development_groups_sha256", "query_sha256", "truth_sha256", "n_evaluated"), "fold")
            require(isinstance(fold["id"], str) and IDENTIFIER.fullmatch(fold["id"]), "invalid fold id")
            require(all(valid_hash(v) for k, v in fold.items() if k not in ("id", "n_evaluated")), "fold hashes required")
            require(integer(fold["n_evaluated"], 1), "freeze each fold's evaluation denominator")
            require(fold["fit_groups_sha256"] != fold["development_groups_sha256"], "identical fit/development hashes")
            ids.append(fold["id"])
        require(len(set(ids)) == len(ids), "duplicate fold ids")
    canonical(plan)
    return copy.deepcopy(plan)


class SearchController:
    def __init__(self, state_dir, plan):
        require(optuna.__version__ == OPTUNA_VERSION, f"use optuna=={OPTUNA_VERSION}")
        require("://" not in str(state_dir), "state must be a local directory, not a remote URL")
        self.plan = validate_plan(plan)
        self.plan_sha256 = digest(self.plan)
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = self.root / "search.sqlite3"
        self.storage = f"sqlite:///{self.db}"

    @contextmanager
    def locked(self):
        fd = os.open(self.root / "controller.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def study(self, objective, trial_number=0):
        require(objective in self.plan["objectives"], "unknown objective")
        # Recreate a public Optuna sampler from a deterministic per-trial seed,
        # not pickle (which can execute code). TPE learns from persisted trials.
        seed = int(digest([self.plan["sampler_seed"], objective, trial_number])[:8], 16)
        sampler = TPESampler(seed=seed, n_startup_trials=self.plan["startup_trials"])
        if trial_number < self.plan["startup_trials"]:
            spec, space = self.plan["objectives"][objective], self.plan["search_space"]
            radius_name = "common_radius_cm" if self.plan["schema"] == CONTEXTUAL_SCHEMA else "radius_cm"
            grid = list(itertools.product(space[radius_name], space["width"], spec["families"]))
            radius, width, family = grid[trial_number]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                sampler = PartialFixedSampler({radius_name: radius, "width": width, "family": family}, sampler)
        # One fixed study name per objective prevents resetting the budget by
        # changing a study/campaign name inside the same state directory.
        study = optuna.create_study(storage=self.storage, study_name=objective,
                                    sampler=sampler, pruner=optuna.pruners.NopPruner(),
                                    direction="minimize", load_if_exists=True)
        require(study.direction.name == "MINIMIZE", "study direction drift")
        attrs = study.user_attrs
        if "plan_sha256" not in attrs:
            require(not study.trials, "unbound study contains trials")
            # Also bind the second objective to the same campaign/plan.
            for other in optuna.get_all_study_summaries(storage=self.storage):
                if other.study_name != objective and "plan_sha256" in other.user_attrs:
                    require(other.user_attrs["plan_sha256"] == self.plan_sha256, "state directory belongs to another plan")
            study.set_user_attr("plan", self.plan)
            study.set_user_attr("plan_sha256", self.plan_sha256)
            study.set_user_attr("optuna_version", OPTUNA_VERSION)
            study.set_user_attr("controller_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        else:
            require(attrs["plan_sha256"] == self.plan_sha256 and attrs["plan"] == self.plan,
                    "frozen plan mismatch; never change search scope after results")
            require(attrs["optuna_version"] == OPTUNA_VERSION, "Optuna version drift")
            require(attrs["controller_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "controller code drift; review before migration")
        return study

    def ask(self, objective):
        with self.locked():
            study = self.study(objective)
            require("frozen_selection" not in study.user_attrs, "selection already frozen")
            trials = study.get_trials(deepcopy=True)
            running = [t for t in trials if t.state == TrialState.RUNNING]
            require(not any(t.state == TrialState.WAITING for t in trials), "unexpected queued trials")
            if running:
                require(len(running) == 1 and "recipe" in running[0].user_attrs,
                        "incomplete proposal transaction; inspect and explicitly fail it")
                return running[0].user_attrs["recipe"]
            require(len(trials) < self.plan["max_recipes"], "recipe budget exhausted")
            number = len(trials)
            if number >= self.plan["startup_trials"]:
                require(all(t.state == TrialState.COMPLETE for t in trials[:self.plan["startup_trials"]]),
                        "startup incomplete: no adaptive proposal or replacement budget is authorized")
            study = self.study(objective, number)
            trial = study.ask()
            require(trial.number == number, "unexpected trial numbering")
            spec, space = self.plan["objectives"][objective], self.plan["search_space"]
            contextual = self.plan["schema"] == CONTEXTUAL_SCHEMA
            radius_name = "common_radius_cm" if contextual else "radius_cm"
            params = {
                "family": trial.suggest_categorical("family", spec["families"]),
                radius_name: trial.suggest_categorical(radius_name, space[radius_name]),
                "width": trial.suggest_categorical("width", space["width"]),
                "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"]["low"],
                                                    space["learning_rate"]["high"], log=True),
            }
            if contextual:
                params.update({
                    "depth": trial.suggest_categorical("depth", space["depth"]),
                    "dropout": trial.suggest_categorical("dropout", space["dropout"]),
                    "rare_radius_cm": (None if params["family"] == "structure_deep_sets" else
                                       trial.suggest_categorical("rare_radius_cm", space["rare_radius_cm"])),
                    "heads": (trial.suggest_categorical("heads", space["heads"])
                              if params["family"].endswith("_local_attention") else None),
                })
            config = {**spec["fixed_config"], **params}
            tasks = []
            for fold, seed, arm in itertools.product(spec["folds"], self.plan["training_seeds"], ARMS):
                tasks.append({"fold": fold["id"], "seed": seed, "arm": arm,
                              "fold_contract": fold, "input_sha256": spec["arm_input_sha256"][arm],
                              "config_sha256": digest(config), "checkpoint_step": spec["checkpoint_step"],
                              "checkpoint_rule": "fixed_updates", "n_evaluated": fold["n_evaluated"],
                              "evaluation_unit": spec["evaluation_unit"]})
            recipe = {"schema": self.plan["schema"], "campaign_id": self.plan["campaign_id"], "objective": objective,
                      "trial_number": number, "phase": "STARTUP" if number < self.plan["startup_trials"] else "TPE",
                      "scope": "DEVELOPMENT", "plan_sha256": self.plan_sha256,
                      "provenance": self.plan["provenance"], "config": config, "tasks": tasks,
                      "primary_metric": spec["primary_metric"], "secondary_metrics": spec["secondary_metrics"],
                      "selection_loss": "equal_fold_seed_arm_mean_absolute_loss", "pruning": "none"}
            recipe["recipe_sha256"] = digest(recipe)
            trial.set_user_attr("recipe", recipe)
            return copy.deepcopy(recipe)

    def validate_receipt(self, receipt, recipe):
        keys(receipt, ("schema", "scope", "objective", "trial_number", "recipe_sha256",
                       "plan_sha256", "provenance", "records"), "receipt")
        for key in ("schema", "scope", "objective", "trial_number", "recipe_sha256", "plan_sha256", "provenance"):
            require(canonical(receipt[key]) == canonical(recipe[key]), f"receipt {key} does not match the proposal")
        require(receipt["scope"] == "DEVELOPMENT", "only development metrics accepted")
        expected = {(task["fold"], task["seed"], task["arm"]): task for task in recipe["tasks"]}
        records = receipt["records"]
        require(isinstance(records, list) and len(records) == len(expected), "complete paired folds/seeds/arms required")
        seen, groups = set(), {}
        names = [recipe["primary_metric"], *recipe["secondary_metrics"]]
        for row in records:
            keys(row, (*next(iter(expected.values())).keys(), "status", "scope", "metrics",
                       "checkpoint_sha256", "prediction_sha256", "n_evaluated", "evaluation_unit"), "metric record")
            key = (row["fold"], row["seed"], row["arm"])
            require(key in expected and key not in seen, "unexpected or duplicate fold/seed/arm")
            seen.add(key)
            require(all(canonical(row[k]) == canonical(v) for k, v in expected[key].items()),
                    "unpaired configuration, fold, seed, input or checkpoint")
            require(row["status"] == "COMPLETE" and row["scope"] == "DEVELOPMENT", "incomplete or non-development result")
            require(valid_hash(row["checkpoint_sha256"]) and valid_hash(row["prediction_sha256"]), "artifact digests required")
            require(integer(row["n_evaluated"], 1), "positive evaluation denominator required")
            require(row["evaluation_unit"] in ("person_macro", "family_macro", "donor_macro", "component_macro"),
                    "declare the dependence-aware macro evaluation unit")
            keys(row["metrics"], names, "metrics")
            require(all(finite(v) for v in row["metrics"].values()), "loss metrics must be finite nonnegative numbers")
            boundary_loss = row["metrics"].get("one_minus_boundary_f1_0p2cm")
            if boundary_loss is not None:
                require(boundary_loss <= 1, "one_minus_boundary_f1_0p2cm must be in [0, 1]")
            groups.setdefault((row["fold"], row["seed"]), {})[row["arm"]] = row
        for paired in groups.values():
            require(paired["NONE"]["n_evaluated"] == paired["RARE"]["n_evaluated"]
                    and paired["NONE"]["evaluation_unit"] == paired["RARE"]["evaluation_unit"],
                    "paired evaluation populations/units differ")
        means = {arm: {name: math.fsum(row["metrics"][name] for row in records if row["arm"] == arm)
                      / len(groups) for name in names} for arm in ARMS}
        primary = recipe["primary_metric"]
        value = (means["NONE"][primary] + means["RARE"][primary]) / 2
        require(math.isfinite(value), "aggregated loss overflow")
        return {"selection_loss": value, "arm_macro_losses": means,
                "delta_rare_minus_none": {name: means["RARE"][name] - means["NONE"][name] for name in names},
                "paired_groups": len(groups), "records": len(records), "scope": "DEVELOPMENT_EXPLORATORY"}

    def tell(self, objective, receipt_path, receipt_sha256):
        receipt, observed = load_json(receipt_path, receipt_sha256)
        with self.locked():
            study = self.study(objective)
            require(receipt.get("objective") == objective, "wrong objective")
            number = receipt.get("trial_number")
            trials = study.get_trials(deepcopy=True)
            require(integer(number) and number < len(trials), "unknown trial")
            trial = trials[number]
            recipe = trial.user_attrs.get("recipe")
            require(recipe is not None, "no committed proposal")
            summary = self.validate_receipt(receipt, recipe)
            if trial.state == TrialState.COMPLETE:
                entry = study.user_attrs.get("receipts", {}).get(str(number), {})
                require(entry.get("sha256") == observed, "completed trial cannot be overwritten")
                return entry["summary"]
            require(trial.state == TrialState.RUNNING, "trial is not running")
            # Public Trial API can be obtained only through ask; attaching attrs
            # via storage would require private IDs. Complete via tell, storing
            # an authenticated journal entry on the study first for crash recovery.
            journal = study.user_attrs.get("receipts", {})
            entry = {"sha256": observed, "receipt": receipt, "summary": summary}
            if str(number) in journal:
                require(journal[str(number)] == entry, "receipt journal is immutable")
            journal[str(number)] = entry
            study.set_user_attr("receipts", journal)
            study.tell(number, summary["selection_loss"])
            return summary

    def fail(self, objective, number, reason):
        require(isinstance(reason, str) and len(reason.strip()) >= 10, "record the operational failure reason")
        with self.locked():
            study = self.study(objective)
            trials = study.get_trials(deepcopy=True)
            require(integer(number) and number < len(trials) and trials[number].state == TrialState.RUNNING,
                    "only a pending trial may fail")
            require(str(number) not in study.user_attrs.get("receipts", {}), "a validated receipt is journalled; complete it instead")
            failures = study.user_attrs.get("failures", {})
            failures[str(number)] = reason
            study.set_user_attr("failures", failures)
            study.tell(number, state=TrialState.FAIL)
            return {"trial_number": number, "status": "FAIL", "counts_against_budget": True}

    def status(self, objective):
        with self.locked():
            study = self.study(objective)
            trials = study.get_trials(deepcopy=True)
            entries = study.user_attrs.get("receipts", {})
            return {"schema": self.plan["schema"], "objective": objective, "plan_sha256": self.plan_sha256,
                    "issued": len(trials), "budget": self.plan["max_recipes"],
                    "trials": [{"number": t.number, "state": t.state.name, "params": t.params,
                                "value": t.value, "summary": entries.get(str(t.number), {}).get("summary")}
                               for t in trials],
                    "selection": study.user_attrs.get("frozen_selection"),
                    "training_integration": "NOT_IMPLEMENTED", "pruning": "none"}

    def freeze(self, objective):
        """Freeze a development choice, never refit or access final evaluation."""
        with self.locked():
            study = self.study(objective)
            trials = study.get_trials(deepcopy=True)
            require(len(trials) == self.plan["max_recipes"] and all(t.state.is_finished() for t in trials),
                    "finish the declared recipe budget before freezing selection")
            require(any(t.state == TrialState.COMPLETE for t in trials), "no completed recipe")
            best = study.best_trial
            selection = {"schema": self.plan["schema"], "objective": objective, "plan_sha256": self.plan_sha256,
                         "selected_trial": best.number, "recipe": best.user_attrs["recipe"],
                         "selection_loss": best.value, "scope": "DEVELOPMENT_SELECTION_NOT_CONFIRMATORY",
                         "refit": "SEPARATE_PLAN_AND_AUTHORIZATION_REQUIRED",
                         "final_evaluation": "NOT_ACCESSED", "training_launched": False}
            existing = study.user_attrs.get("frozen_selection")
            require(existing is None or existing == selection, "frozen selection mismatch")
            study.set_user_attr("frozen_selection", selection)
            return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--objective", choices=("structure", "lai"), required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("ask", "status", "freeze"):
        commands.add_parser(command)
    tell = commands.add_parser("tell")
    tell.add_argument("--receipt", type=Path, required=True)
    tell.add_argument("--receipt-sha256", required=True)
    fail = commands.add_parser("fail")
    fail.add_argument("--trial", type=int, required=True)
    fail.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        plan, _ = load_json(args.plan)
        controller = SearchController(args.state_dir, plan)
        if args.command == "tell":
            result = controller.tell(args.objective, args.receipt, args.receipt_sha256)
        elif args.command == "fail":
            result = controller.fail(args.objective, args.trial, args.reason)
        else:
            result = getattr(controller, args.command)(args.objective)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"search contract rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
