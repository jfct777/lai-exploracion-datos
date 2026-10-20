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
CONTROLLED_SCHEMA = "genomic-adaptive-search/3"
CONTEXT_POLICY_SCHEMA = "structure_search_context_policy_v1"
K03_PRIMARY_METRIC = "k03_component_pair_weighted_mse"
K03_EVALUATION_UNIT = "authenticated_k03_component_pair_weights"
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
CONTROLLED_ARMS = ("NONE", "COUNTS", "RARE")
RARE_POOLING_POLICIES = ("SUM", "MEAN_WITH_EXPLICIT_BURDEN")
LOCUS_DESCRIPTORS = ("ALLELE_POSITION_12_V0", "ALLELE_POSITION_CHROM22_V1")
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


def request_context(config):
    """Return only fields that determine an autosomal reader request."""
    family = config["family"]
    return {
        "family": family,
        "common_radius_cm": config["common_radius_cm"],
        "rare_radius_cm": config["rare_radius_cm"],
        "attention_depth": 0 if family == "structure_deep_sets" else config["depth"],
    }


def prospective_request_contexts(plan, objective):
    """Enumerate frozen request shapes without enumerating learning rates/widths."""
    if objective != "structure" or "context_policy" not in plan:
        return []
    space = plan["search_space"]
    contexts = []
    for family, radius in itertools.product(plan["objectives"][objective]["families"],
                                             space["common_radius_cm"]):
        config = {
            "family": family,
            "common_radius_cm": radius,
            "rare_radius_cm": None if family == "structure_deep_sets" else radius,
            "depth": plan["context_policy"]["fixed"]["depth"],
        }
        context = request_context(config)
        contexts.append({"sha256": digest(context), "request": context})
    require(len({row["sha256"] for row in contexts}) == len(contexts),
            "prospective request contexts are not unique")
    return contexts


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
                "provenance", "pruning") + (("actual_data_scope",) if plan.get("schema") == CONTROLLED_SCHEMA else ())
                + (("context_policy",) if "context_policy" in plan else ()), "plan")
    require(plan["schema"] in (SCHEMA, CONTEXTUAL_SCHEMA, CONTROLLED_SCHEMA) and plan["scope"] == "DEVELOPMENT",
            "supported versioned development plan required")
    contextual = plan["schema"] != SCHEMA
    controlled = plan["schema"] == CONTROLLED_SCHEMA
    if controlled:
        require(plan.get("actual_data_scope") in ("SYNTHETIC_CODE_TEST", "DEVELOPMENT"),
                "v3 must freeze the actual data scope, separately from selection stage")
    if "context_policy" in plan:
        require(controlled, "context_policy is supported only by the controlled v3 schema")
        policy = plan["context_policy"]
        keys(policy, ("schema", "attention_rare_radius", "fixed", "selection_design",
                      "rare_effect_selection"), "context_policy")
        require(policy["schema"] == CONTEXT_POLICY_SCHEMA
                and policy["attention_rare_radius"] == "match_common_radius"
                and policy["selection_design"] == "single_frozen_fit_select"
                and policy["rare_effect_selection"] == "report_only",
                "unsupported prospective context policy")
        keys(policy["fixed"], ("depth", "dropout", "heads"), "context_policy.fixed")
        require(policy["fixed"] == {"depth": 2, "dropout": 0.0, "heads": 4},
                "prospective context policy fixed architecture differs")
    arms = CONTROLLED_ARMS if controlled else ARMS
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
    if "context_policy" in plan:
        fixed = plan["context_policy"]["fixed"]
        require(space["rare_radius_cm"] == space["common_radius_cm"],
                "context_policy requires the same prospective common/rare radius choices")
        require(space["depth"] == [fixed["depth"]]
                and space["dropout"] == [fixed["dropout"]]
                and space["heads"] == [fixed["heads"]],
                "context_policy architecture fields must be fixed singleton choices")
        require(plan["max_recipes"] == 24,
                "context_policy freezes the 24-recipe planning ceiling")
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
    if "context_policy" in plan:
        require(set(plan["objectives"]) == {"structure"},
                "structure context_policy cannot be applied to another objective")
    for name, spec in plan["objectives"].items():
        expected = ("families", "primary_metric", "secondary_metrics", "folds", "fixed_config",
                    "checkpoint_step", "arm_input_sha256", "range_rationale", "evaluation_unit")
        keys(spec, (*expected, "harm_limits") if controlled else expected, name)
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
        rare_pooling = spec["fixed_config"].get("rare_pooling", "SUM")
        require(rare_pooling in RARE_POOLING_POLICIES and
                (name == "structure" or rare_pooling == "SUM"),
                "rare_pooling must be a supported fixed structure option")
        locus_descriptor = spec["fixed_config"].get("locus_descriptor", "ALLELE_POSITION_12_V0")
        require(locus_descriptor in LOCUS_DESCRIPTORS and
                (name == "structure" or locus_descriptor == "ALLELE_POSITION_12_V0"),
                "locus_descriptor must be a supported fixed structure option")
        if "context_policy" in plan:
            require(spec["fixed_config"].get("optimizer") == "Adam"
                    and spec["fixed_config"].get("adam_state_precision") in ("fp32", "fp64"),
                    "prospective context search must bind Adam and its state precision")
        reserved = {"family", "radius_cm", "width", "learning_rate", "seed", "arm", "fold"}
        if contextual:
            reserved |= CONTEXTUAL_FIELDS
        require(not set(spec["fixed_config"]) & reserved,
                "fixed config must not override sampled or paired fields")
        require(integer(spec["checkpoint_step"], 1), "positive fixed checkpoint step required")
        require(spec["evaluation_unit"] in ("person_macro", "family_macro", "donor_macro", "component_macro",
                                             K03_EVALUATION_UNIT),
                "freeze a dependence-aware evaluation unit per objective")
        if spec["primary_metric"] == K03_PRIMARY_METRIC or spec["evaluation_unit"] == K03_EVALUATION_UNIT:
            require(controlled and name == "structure"
                    and spec["primary_metric"] == K03_PRIMARY_METRIC
                    and spec["evaluation_unit"] == K03_EVALUATION_UNIT
                    and not spec["secondary_metrics"] and "context_policy" in plan,
                    "K03 weighted loss and authenticated weights must be paired in a prospective controlled structure plan")
        require(isinstance(spec["range_rationale"], str) and len(spec["range_rationale"].strip()) >= 20,
                "record the dataset-specific rationale; ranges are not established optima")
        keys(spec["arm_input_sha256"], arms, "arm input manifests")
        require(all(valid_hash(v) for v in spec["arm_input_sha256"].values()), "arm input hashes required")
        require(spec["arm_input_sha256"]["NONE"] != spec["arm_input_sha256"]["RARE"], "distinct arm manifests required")
        if controlled:
            require(len(set(spec["arm_input_sha256"].values())) == len(arms), "distinct arm manifests required")
            require(isinstance(spec["harm_limits"], dict) and spec["harm_limits"],
                    "v3 requires explicit development harm limits; no scientific tolerance defaults")
            require(set(spec["harm_limits"]) <= set(metrics)
                    and all(finite(v) for v in spec["harm_limits"].values()), "invalid harm limits")
        folds = spec["folds"]
        single_split = plan.get("context_policy", {}).get("selection_design") == "single_frozen_fit_select"
        require(isinstance(folds, list) and (len(folds) == 1 if single_split else len(folds) >= 2),
                "development fold count differs from the frozen selection design")
        ids = []
        for fold in folds:
            keys(fold, ("id", "fit_groups_sha256", "development_groups_sha256", "query_sha256",
                        "truth_sha256", "n_evaluated"), "fold")
            require(isinstance(fold["id"], str) and IDENTIFIER.fullmatch(fold["id"]), "invalid fold id")
            require(all(valid_hash(v) for k, v in fold.items() if k not in ("id", "n_evaluated")), "fold hashes required")
            require(integer(fold["n_evaluated"], 1), "freeze each fold's evaluation denominator")
            require(fold["fit_groups_sha256"] != fold["development_groups_sha256"], "identical fit/development hashes")
            ids.append(fold["id"])
        require(len(set(ids)) == len(ids), "duplicate fold ids")
    if "context_policy" in plan:
        require(plan["startup_trials"] == len(prospective_request_contexts(plan, "structure")) * len(space["width"]),
                "context_policy startup must cover each family/radius/width recipe once")
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
            radius_name = "common_radius_cm" if self.plan["schema"] != SCHEMA else "radius_cm"
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
            contextual = self.plan["schema"] != SCHEMA
            radius_name = "common_radius_cm" if contextual else "radius_cm"
            params = {
                "family": trial.suggest_categorical("family", spec["families"]),
                radius_name: trial.suggest_categorical(radius_name, space[radius_name]),
                "width": trial.suggest_categorical("width", space["width"]),
                "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"]["low"],
                                                    space["learning_rate"]["high"], log=True),
            }
            if contextual:
                if "context_policy" in self.plan:
                    fixed = self.plan["context_policy"]["fixed"]
                    params.update({
                        "depth": fixed["depth"], "dropout": fixed["dropout"],
                        "rare_radius_cm": (None if params["family"] == "structure_deep_sets"
                                           else params["common_radius_cm"]),
                        "heads": (fixed["heads"] if params["family"].endswith("_local_attention") else None),
                    })
                else:
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
            arms = CONTROLLED_ARMS if self.plan["schema"] == CONTROLLED_SCHEMA else ARMS
            for fold, seed, arm in itertools.product(spec["folds"], self.plan["training_seeds"], arms):
                tasks.append({"fold": fold["id"], "seed": seed, "arm": arm,
                              "fold_contract": fold, "input_sha256": spec["arm_input_sha256"][arm],
                              "config_sha256": digest(config), "checkpoint_step": spec["checkpoint_step"],
                              "checkpoint_rule": "fixed_updates", "n_evaluated": fold["n_evaluated"],
                              "evaluation_unit": spec["evaluation_unit"],
                              **({"request_context_sha256": digest(request_context(config)),
                                  "dataset_mode": "AUTOSOMAL_PERSON_BATCHES_V1"}
                                 if "context_policy" in self.plan else {})})
            recipe = {"schema": self.plan["schema"], "campaign_id": self.plan["campaign_id"], "objective": objective,
                      "trial_number": number, "phase": "STARTUP" if number < self.plan["startup_trials"] else "TPE",
                      "scope": "DEVELOPMENT", "plan_sha256": self.plan_sha256,
                      "provenance": self.plan["provenance"], "config": config, "tasks": tasks,
                      "primary_metric": spec["primary_metric"], "secondary_metrics": spec["secondary_metrics"],
                      "selection_loss": ("single_frozen_fit_select_seed_arm_mean_loss"
                                         if "context_policy" in self.plan else
                                         "equal_fold_seed_arm_mean_absolute_loss"),
                      "pruning": "none"}
            if self.plan["schema"] == CONTROLLED_SCHEMA:
                recipe["actual_data_scope"] = self.plan["actual_data_scope"]
            if "context_policy" in self.plan:
                recipe["selection_design"] = self.plan["context_policy"]["selection_design"]
            recipe["recipe_sha256"] = digest(recipe)
            trial.set_user_attr("recipe", recipe)
            return copy.deepcopy(recipe)

    def validate_receipt(self, receipt, recipe):
        keys(receipt, ("schema", "scope", "objective", "trial_number", "recipe_sha256",
                       "plan_sha256", "provenance", "records")
                      + (("actual_data_scope",) if self.plan["schema"] == CONTROLLED_SCHEMA else ())
                      + (("selection_design",) if "context_policy" in self.plan else ()), "receipt")
        for key in (("schema", "scope", "objective", "trial_number", "recipe_sha256", "plan_sha256", "provenance")
                    + (("selection_design",) if "context_policy" in self.plan else ())):
            require(canonical(receipt[key]) == canonical(recipe[key]), f"receipt {key} does not match the proposal")
        require(receipt["scope"] == "DEVELOPMENT", "only development metrics accepted")
        if self.plan["schema"] == CONTROLLED_SCHEMA:
            require(receipt.get("actual_data_scope") == recipe["actual_data_scope"] == self.plan["actual_data_scope"],
                    "receipt actual data scope differs from frozen plan")
        expected = {(task["fold"], task["seed"], task["arm"]): task for task in recipe["tasks"]}
        records = receipt["records"]
        require(isinstance(records, list) and len(records) == len(expected), "complete paired folds/seeds/arms required")
        seen, groups = set(), {}
        names = [recipe["primary_metric"], *recipe["secondary_metrics"]]
        for row in records:
            audit_fields = (("contract_sha256", "resolved_sha256", "training_receipt_sha256",
                             "checkpoint_identity_sha256", "restored_steps", "adam_state_precision",
                             "rare_pooling")
                            if "context_policy" in self.plan else ())
            keys(row, (*next(iter(expected.values())).keys(), "status", "scope", "metrics",
                       "checkpoint_sha256", "prediction_sha256", "n_evaluated", "evaluation_unit",
                       *audit_fields), "metric record")
            key = (row["fold"], row["seed"], row["arm"])
            require(key in expected and key not in seen, "unexpected or duplicate fold/seed/arm")
            seen.add(key)
            require(all(canonical(row[k]) == canonical(v) for k, v in expected[key].items()),
                    "unpaired configuration, fold, seed, input or checkpoint")
            require(row["status"] == "COMPLETE" and row["scope"] == "DEVELOPMENT", "incomplete or non-development result")
            require(valid_hash(row["checkpoint_sha256"]) and valid_hash(row["prediction_sha256"]), "artifact digests required")
            require(integer(row["n_evaluated"], 1), "positive evaluation denominator required")
            require(row["evaluation_unit"] in ("person_macro", "family_macro", "donor_macro", "component_macro",
                                                K03_EVALUATION_UNIT),
                    "declare the dependence-aware macro evaluation unit")
            if audit_fields:
                require(all(valid_hash(row[field]) for field in audit_fields[:4])
                        and integer(row["restored_steps"])
                        and row["adam_state_precision"] == recipe["config"]["adam_state_precision"]
                        and row["rare_pooling"] == recipe["config"].get("rare_pooling", "SUM")
                        and row["dataset_mode"] == "AUTOSOMAL_PERSON_BATCHES_V1",
                        "authenticated training/checkpoint identities differ")
            keys(row["metrics"], names, "metrics")
            require(all(finite(v) for v in row["metrics"].values()), "loss metrics must be finite nonnegative numbers")
            boundary_loss = row["metrics"].get("one_minus_boundary_f1_0p2cm")
            if boundary_loss is not None:
                require(boundary_loss <= 1, "one_minus_boundary_f1_0p2cm must be in [0, 1]")
            groups.setdefault((row["fold"], row["seed"]), {})[row["arm"]] = row
        for paired in groups.values():
            arms = CONTROLLED_ARMS if self.plan["schema"] == CONTROLLED_SCHEMA else ARMS
            require(set(paired) == set(arms)
                    and len({paired[arm]["n_evaluated"] for arm in arms}) == 1
                    and len({paired[arm]["evaluation_unit"] for arm in arms}) == 1,
                    "paired evaluation populations/units differ")
        arms = CONTROLLED_ARMS if self.plan["schema"] == CONTROLLED_SCHEMA else ARMS
        means = {arm: {name: math.fsum(row["metrics"][name] for row in records if row["arm"] == arm)
                      / len(groups) for name in names} for arm in arms}
        primary = recipe["primary_metric"]
        value = math.fsum(means[arm][primary] for arm in arms) / len(arms)
        require(math.isfinite(value), "aggregated loss overflow")
        summary = {"selection_loss": value, "arm_macro_losses": means,
                "delta_rare_minus_none": {name: means["RARE"][name] - means["NONE"][name] for name in names},
                "paired_groups": len(groups), "records": len(records), "scope": "DEVELOPMENT_EXPLORATORY"}
        if self.plan["schema"] == CONTROLLED_SCHEMA:
            summary["actual_data_scope"] = self.plan["actual_data_scope"]
            limits = self.plan["objectives"][recipe["objective"]]["harm_limits"]
            summary["delta_rare_minus_counts"] = {name: means["RARE"][name] - means["COUNTS"][name] for name in names}
            summary["development_harm_checks"] = {
                comparator: {name: means["RARE"][name] - means[comparator][name] <= limit
                             for name, limit in limits.items()}
                for comparator in ("NONE", "COUNTS")}
            if "context_policy" in self.plan:
                # For the prospective structure search, rare-effect differences
                # describe the shared recipe. They do not select recipes or tune
                # each arm independently.
                summary["rare_effect_selection"] = self.plan["context_policy"]["rare_effect_selection"]
                summary["eligible_development_selection"] = True
            else:
                summary["eligible_development_selection"] = all(
                    all(checks.values()) for checks in summary["development_harm_checks"].values())
            summary["harm_check_scope"] = "point_estimates_only_not_confirmatory_uncertainty"
        return summary

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
                    "training_integration": ("BOUNDED_THREE_ARM_ADAPTER_AVAILABLE" if self.plan["schema"] == CONTROLLED_SCHEMA else "NOT_IMPLEMENTED"), "pruning": "none"}

    def freeze(self, objective):
        """Freeze a development choice, never refit or access final evaluation."""
        with self.locked():
            study = self.study(objective)
            trials = study.get_trials(deepcopy=True)
            require(len(trials) == self.plan["max_recipes"] and all(t.state.is_finished() for t in trials),
                    "finish the declared recipe budget before freezing selection")
            require(any(t.state == TrialState.COMPLETE for t in trials), "no completed recipe")
            eligible = [t for t in trials if t.state == TrialState.COMPLETE]
            if self.plan["schema"] == CONTROLLED_SCHEMA and "context_policy" not in self.plan:
                journal = study.user_attrs.get("receipts", {})
                eligible = [t for t in eligible if journal[str(t.number)]["summary"]["eligible_development_selection"]]
                require(eligible, "no recipe satisfies the prespecified development harm limits")
            best = min(eligible, key=lambda t: (t.value, t.number))
            selection = {"schema": self.plan["schema"], "objective": objective, "plan_sha256": self.plan_sha256,
                         "selected_trial": best.number, "recipe": best.user_attrs["recipe"],
                         "selection_loss": best.value, "scope": "DEVELOPMENT_SELECTION_NOT_CONFIRMATORY",
                         "refit": "SEPARATE_PLAN_AND_AUTHORIZATION_REQUIRED",
                         "final_evaluation": "NOT_ACCESSED", "training_launched": False}
            if self.plan["schema"] == CONTROLLED_SCHEMA:
                selection["actual_data_scope"] = self.plan["actual_data_scope"]
                if "context_policy" in self.plan:
                    journal = study.user_attrs.get("receipts", {})
                    require(all(journal[str(t.number)]["summary"].get("rare_effect_selection") == "report_only"
                                and journal[str(t.number)]["summary"]["eligible_development_selection"] is True
                                for t in eligible),
                            "prospective rare-effect reporting policy differs from completed receipts")
                    selection["rare_effect_selection"] = "report_only"
                if self.plan["actual_data_scope"] == "SYNTHETIC_CODE_TEST":
                    selection["scope"] = "SYNTHETIC_CODE_TEST_ONLY"
                    selection["refit"] = "SCIENTIFIC_REUSE_FORBIDDEN"
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
