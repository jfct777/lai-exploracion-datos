#!/usr/bin/env python3
"""Execute one complete Optuna proposal through authenticated trainer tasks.

The existing controller owns proposals/budget; this bridge never invents
metrics, roles or tolerances. Every fold uses one dataset for all three arms.
Nextflow invokes ``prepare``, each ``task``, then ``collect``; a completed task
can be cached. Final evaluation and automatic search-domain expansion are absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import genomic_adaptive_search as search
from r02_context_train import (AUTOSOMAL_POOL, SINGLE_CHROMOSOME, dataset_mode,
    load_batch, read_json, sha, train, validate_contract, validate_training_entry, write_json)


def validate_inputs(plan_path, binding_path, binding_sha256, objective):
    """Reauthenticate actual tensors before Nextflow may reuse any task."""
    plan, _ = search.load_json(plan_path)
    search.require(plan["schema"] == search.CONTROLLED_SCHEMA, "bridge requires three-arm v3")
    binding = read_json(binding_path, binding_sha256)
    search.require(binding["schema"] == "r02_context_search_binding_v1", "unsupported binding")
    search.require(binding["plan_sha256"] == search.digest(plan), "binding belongs to another plan")
    search.require(plan.get("actual_data_scope") == binding["scope"], "plan/binding actual data scope differs")
    spec = plan["objectives"][objective]
    search.require(spec["primary_metric"] in ("component_mse", "component_log_loss")
                   and not spec["secondary_metrics"],
                   "this bounded trainer has no dense LAI/border adapter; do not substitute its selection metric")
    contracts = {}
    # Validate BEFORE asking, so an input mistake cannot consume recipe budget.
    for fold in spec["folds"]:
        descriptor = binding["folds"][fold["id"]]
        contract = validate_contract(read_json(descriptor["path"], descriptor["sha256"]))
        if binding["scope"] == "DEVELOPMENT":
            expected_scope = ("AUTOSOMAL_22" if dataset_mode(contract) == AUTOSOMAL_POOL
                              else "BOUNDED_CONTEXT_PILOT")
            search.require(binding.get("dataset_declared_scope") == contract["dataset_declared_scope"] == expected_scope,
                           "binding declared data scope differs from the explicit trainer mode")
        search.require(dataset_mode(contract) in (SINGLE_CHROMOSOME, AUTOSOMAL_POOL), "unsupported trainer data mode")
        for stage, batches in contract["batches"].items():
            for entry in batches:
                # Resolved radii are proposal-specific and are authenticated by
                # each task. Preflight still validates all fixed target tensors.
                if dataset_mode(contract) == SINGLE_CHROMOSOME:
                    validate_training_entry(entry, contract, stage)
                else:
                    load_batch(entry, contract, stage)
        search.require(contract["objective"] == objective and contract["steps"] == spec["checkpoint_step"], "task objective/budget mismatch")
        search.require(contract["scope"] == binding["scope"] and binding["scope"] in ("DEVELOPMENT", "SYNTHETIC_CODE_TEST"), "scope mismatch")
        groups = {stage: sorted({p["component"] for p in contract["persons"].values() if p["role"] == stage})
                  for stage in ("FIT", "SELECT")}
        search.require(search.digest(groups["FIT"]) == fold["fit_groups_sha256"]
                       and search.digest(groups["SELECT"]) == fold["development_groups_sha256"], "fold groups mismatch")
        search.require(len(groups["SELECT"]) == fold["n_evaluated"] and spec["evaluation_unit"] == "component_macro", "component denominator mismatch")
        search.require(contract["query_sha256"] == fold["query_sha256"]
                       and contract["truth_sha256"] == fold["truth_sha256"], "fold target binding mismatch")
        contracts[fold["id"]] = descriptor
    for arm in search.CONTROLLED_ARMS:
        search.require(search.digest({"arm": arm, "folds": contracts}) == spec["arm_input_sha256"][arm], "arm input binding mismatch")
    search.require(plan["provenance"]["trainer_code_sha256"] == sha(Path(__file__).with_name("r02_context_train.py")), "trainer source drift")
    search.require(binding["runtime_sha256"] == sha(Path(__file__).with_name("genomic_context_runtime.py")), "runtime source drift")
    search.require(plan["provenance"]["dataset_sha256"] == search.digest({"folds": contracts, "scope": binding["scope"]}), "dataset/scope binding mismatch")
    return plan, binding, contracts


def prepare(plan_path, binding_path, binding_sha256, objective, state_dir, output, trial_number):
    plan, binding, contracts = validate_inputs(plan_path, binding_path, binding_sha256, objective)
    controller = search.SearchController(state_dir, plan)
    with controller.locked():
        study = controller.study(objective)
        trials = study.get_trials(deepcopy=True)
        search.require(type(trial_number) is int and 0 <= trial_number <= len(trials), "unexpected requested trial")
        prior = trials[trial_number].user_attrs.get("recipe") if trial_number < len(trials) else None
    recipe = prior if prior is not None else controller.ask(objective)
    search.require(recipe["trial_number"] == trial_number, "previous recipe still pending; never bypass it")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "recipe.json", recipe)
    write_json(output / "resolved.json", recipe["config"])
    tasks = []
    for index, task in enumerate(recipe["tasks"]):
        record = {"recipe": recipe, "task": task, "contract": contracts[task["fold"]],
                  "scope": binding["scope"], "runtime_sha256": binding["runtime_sha256"],
                  "resolved_sha256": sha(output / "resolved.json")}
        path = output / f"task-{index:04d}.json"
        write_json(path, record)
        tasks.append({"path": str(path.resolve()), "sha256": sha(path)})
    write_json(output / "tasks.json", tasks)
    return tasks


def execute(task_path, task_sha256, resolved_path, output, checkpoint_root=None,
            checkpoint_every_steps=1, adam_state_precision="fp32"):
    record = read_json(task_path, task_sha256)
    task = record["task"]
    search.require(record["recipe"]["provenance"]["trainer_code_sha256"] == sha(Path(__file__).with_name("r02_context_train.py")), "trainer source drift")
    search.require(record["runtime_sha256"] == sha(Path(__file__).with_name("genomic_context_runtime.py")), "runtime source drift")
    checkpoint_dir = ((Path(checkpoint_root) / task_sha256) if checkpoint_root is not None
                      else Path(output).parent / f".{Path(output).name}.{task_sha256}.checkpoints")
    receipt = train(record["contract"]["path"], record["contract"]["sha256"], resolved_path,
                    record["resolved_sha256"], task["arm"], task["seed"], output,
                    checkpoint_dir, checkpoint_every_steps, adam_state_precision)
    search.require(receipt["scope"] == record["scope"] and receipt["steps"] == task["checkpoint_step"], "training scope/budget drift")
    search.require(receipt["n_evaluated"] == task["n_evaluated"]
                   and receipt["evaluation_unit"] == task["evaluation_unit"], "training denominator drift")
    metric = record["recipe"]["primary_metric"]
    search.require(metric in receipt["metrics"], "requested metric not implemented by trainer")
    result = {**task, "status": "COMPLETE", "scope": "DEVELOPMENT", "metrics": {metric: receipt["metrics"][metric]},
              "checkpoint_sha256": receipt["checkpoint_sha256"], "prediction_sha256": receipt["prediction_sha256"]}
    write_json(Path(output) / "search-record.json", {"record": result, "actual_scope": receipt["scope"],
        "task_sha256": task_sha256, "training_receipt_sha256": sha(Path(output) / "training.receipt.json")})
    return result


def collect(plan_path, state_dir, recipe_path, task_index_path, result_paths, output):
    plan, _ = search.load_json(plan_path)
    recipe, _ = search.load_json(recipe_path)
    tasks, _ = search.load_json(task_index_path)
    expected = {entry["sha256"]: read_json(entry["path"], entry["sha256"]) for entry in tasks}
    search.require(len(expected) == len(tasks), "duplicate task index")
    records, scopes, seen = [], set(), set()
    for path in result_paths:
        result, _ = search.load_json(path)
        task_hash = result["task_sha256"]
        search.require(task_hash in expected and task_hash not in seen, "unexpected/duplicate task result")
        # Reopen artifacts: a JSON success flag alone is not sufficient.
        parent = Path(path).parent
        training = read_json(parent / "training.receipt.json", result["training_receipt_sha256"])
        source = expected[task_hash]
        search.require(source["recipe"] == recipe, "task belongs to another recipe")
        search.require(all(result["record"][key] == value for key, value in source["task"].items()), "task fields changed")
        search.require(training["arm"] == source["task"]["arm"]
                       and training["seed"] == source["task"]["seed"]
                       and training["steps"] == source["task"]["checkpoint_step"]
                       and training["contract_sha256"] == source["contract"]["sha256"]
                       and training["resolved_sha256"] == source["resolved_sha256"]
                       and training["trainer_sha256"] == plan["provenance"]["trainer_code_sha256"]
                       and training["runtime_sha256"] == source["runtime_sha256"]
                       and training["scope"] == source["scope"] == result["actual_scope"], "training receipt does not belong to this task")
        search.require(sha(parent / "weights.npz") == training["checkpoint_sha256"], "missing/changed checkpoint")
        predictions = read_json(parent / "predictions.json", training["prediction_sha256"])
        for prediction in predictions:
            search.require(sha(prediction["path"]) == prediction["sha256"], "missing/changed prediction")
        search.require(result["record"]["metrics"] == training["metrics"], "metric receipt changed")
        seen.add(task_hash)
        records.append(result["record"])
        scopes.add(result["actual_scope"])
    search.require(seen == set(expected) and len(scopes) == 1, "incomplete/mixed-scope recipe")
    receipt = {key: recipe[key] for key in ("schema", "scope", "objective", "trial_number", "recipe_sha256", "plan_sha256", "provenance")}
    # Nextflow completion order is nondeterministic; an immutable receipt must
    # be byte-identical when the same tasks are collected after -resume.
    receipt["records"] = sorted(records, key=lambda row: (row["fold"], row["seed"], row["arm"]))
    receipt["actual_data_scope"] = next(iter(scopes))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "receipt.json", receipt)
    controller = search.SearchController(state_dir, plan)
    summary = controller.tell(recipe["objective"], output / "receipt.json", sha(output / "receipt.json"))
    summary["actual_data_scope"] = next(iter(scopes))
    summary["scientific_conclusion"] = "NOT_ESTABLISHED"
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    for key in ("plan", "binding", "binding-sha256", "objective", "output"):
        preflight_parser.add_argument("--" + key, required=True)
    prepare_parser = commands.add_parser("prepare")
    for key in ("plan", "binding", "binding-sha256", "objective", "state-dir", "output"):
        prepare_parser.add_argument("--" + key, required=True)
    prepare_parser.add_argument("--trial-number", type=int, required=True)
    task_parser = commands.add_parser("task")
    for key in ("task", "task-sha256", "resolved", "output"):
        task_parser.add_argument("--" + key, required=True)
    task_parser.add_argument("--checkpoint-root")
    task_parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    task_parser.add_argument("--adam-state-precision", choices=("fp32", "fp64"), default="fp32")
    collect_parser = commands.add_parser("collect")
    for key in ("plan", "state-dir", "recipe", "task-index", "output"):
        collect_parser.add_argument("--" + key, required=True)
    collect_parser.add_argument("--results", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        plan, binding, contracts = validate_inputs(args.plan, args.binding, args.binding_sha256, args.objective)
        result = {"status": "AUTHENTICATED", "plan_sha256": search.digest(plan),
                  "binding_sha256": args.binding_sha256, "folds": contracts}
        write_json(args.output, result)
    elif args.command == "prepare":
        result = prepare(args.plan, args.binding, args.binding_sha256, args.objective, args.state_dir, args.output, args.trial_number)
    elif args.command == "task":
        result = execute(args.task, args.task_sha256, args.resolved, args.output,
                         args.checkpoint_root, args.checkpoint_every_steps,
                         args.adam_state_precision)
    else:
        result = collect(args.plan, args.state_dir, args.recipe, args.task_index, args.results, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
