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
from genomic_context_runtime import LOCUS_DESCRIPTOR_LEGACY
from r02_context_autosomal_pool import RARE_POOLING_SUM
from r02_context_train import (AUTOSOMAL_PERSON_BATCHES, AUTOSOMAL_POOL, SINGLE_CHROMOSOME,
    dataset_mode, load_batch, read_json, sha, train, validate_contract, validate_training_entry, write_json)


BINDING_V1 = "r02_context_search_binding_v1"
BINDING_V2 = "r02_context_search_binding_v2"


def resolved_template(plan, objective, request):
    """Build one valid recipe solely to authenticate a frozen reader request."""
    spec, policy, space = plan["objectives"][objective], plan["context_policy"], plan["search_space"]
    family = request["family"]
    config = {**spec["fixed_config"], "family": family,
              "common_radius_cm": request["common_radius_cm"],
              "rare_radius_cm": request["rare_radius_cm"],
              "width": space["width"][0], "learning_rate": space["learning_rate"]["low"],
              "depth": policy["fixed"]["depth"], "dropout": policy["fixed"]["dropout"],
              "heads": policy["fixed"]["heads"] if family.endswith("_local_attention") else None}
    search.require(search.request_context(config) == request,
                   "binding request cannot be represented by the prospective search policy")
    return config


def validate_fold_contract(contract, descriptor, plan, binding, objective, fold, resolved=None):
    """Authenticate one exact contract and the target/person axes it declares."""
    spec = plan["objectives"][objective]
    mode = dataset_mode(contract)
    if binding["scope"] == "DEVELOPMENT":
        expected_scope = ("AUTOSOMAL_22" if mode in (AUTOSOMAL_POOL, AUTOSOMAL_PERSON_BATCHES)
                          else "BOUNDED_CONTEXT_PILOT")
        search.require(binding.get("dataset_declared_scope") == contract["dataset_declared_scope"] == expected_scope,
                       "binding declared data scope differs from the explicit trainer mode")
    search.require(mode in (SINGLE_CHROMOSOME, AUTOSOMAL_POOL, AUTOSOMAL_PERSON_BATCHES),
                   "unsupported trainer data mode")
    for stage, batches in contract["batches"].items():
        for entry in batches:
            # The trainer function selects load_batch or the real autosomal reader,
            # authenticates the request/target axes and enforces resolved halos.
            if mode == AUTOSOMAL_POOL and resolved is None:
                load_batch(entry, contract, stage)
            else:
                validate_training_entry(entry, contract, stage, resolved)
    search.require(contract["objective"] == objective and contract["steps"] == spec["checkpoint_step"],
                   "task objective/budget mismatch")
    search.require(contract["scope"] == binding["scope"] and binding["scope"] in
                   ("DEVELOPMENT", "SYNTHETIC_CODE_TEST"), "scope mismatch")
    groups = {stage: sorted({p["component"] for p in contract["persons"].values() if p["role"] == stage})
              for stage in ("FIT", "SELECT")}
    search.require(search.digest(groups["FIT"]) == fold["fit_groups_sha256"]
                   and search.digest(groups["SELECT"]) == fold["development_groups_sha256"],
                   "fold groups mismatch")
    search.require(len(groups["SELECT"]) == fold["n_evaluated"], "component denominator mismatch")
    search.require(contract["query_sha256"] == fold["query_sha256"]
                   and contract["truth_sha256"] == fold["truth_sha256"], "fold target binding mismatch")
    return {"descriptor": descriptor, "mode": mode, "groups": groups}


def validate_inputs(plan_path, binding_path, binding_sha256, objective):
    """Reauthenticate actual tensors before Nextflow may reuse any task."""
    plan, _ = search.load_json(plan_path)
    plan = search.validate_plan(plan)
    search.require(plan["schema"] == search.CONTROLLED_SCHEMA, "bridge requires three-arm v3")
    binding = read_json(binding_path, binding_sha256)
    search.require(binding["schema"] in (BINDING_V1, BINDING_V2), "unsupported binding")
    binding_payload = (("schema", "plan_sha256", "scope", "dataset_declared_scope",
                        "runtime_sha256", "contexts") if binding["schema"] == BINDING_V2 else
                       ("schema", "plan_sha256", "scope", "dataset_declared_scope",
                        "runtime_sha256", "folds"))
    search.keys(binding, binding_payload, "search binding")
    search.require(binding["plan_sha256"] == search.digest(plan), "binding belongs to another plan")
    search.require(plan.get("actual_data_scope") == binding["scope"], "plan/binding actual data scope differs")
    spec = plan["objectives"][objective]
    prospective = "context_policy" in plan
    if prospective:
        search.require(binding["schema"] == BINDING_V2 and objective == "structure"
                       and spec["primary_metric"] == search.K03_PRIMARY_METRIC
                       and spec["evaluation_unit"] == search.K03_EVALUATION_UNIT
                       and not spec["secondary_metrics"],
                       "prospective person-batch search requires the reviewed K03 metric/unit and v2 binding")
        expected_contexts = {row["sha256"]: row["request"]
                             for row in search.prospective_request_contexts(plan, objective)}
        search.require(set(binding.get("contexts", {})) == set(expected_contexts),
                       "binding must resolve every and only prospective request context")
        contexts = {}
        for context_sha256, expected_request in expected_contexts.items():
            row = binding["contexts"][context_sha256]
            search.keys(row, ("request", "folds"), "binding context")
            search.require(row["request"] == expected_request
                           and search.digest(row["request"]) == context_sha256,
                           "binding request context/hash differs")
            search.require(set(row["folds"]) == {fold["id"] for fold in spec["folds"]},
                           "binding context fold coverage differs")
            resolved = resolved_template(plan, objective, expected_request)
            descriptors = {}
            for fold in spec["folds"]:
                descriptor = row["folds"][fold["id"]]
                search.keys(descriptor, ("path", "sha256"), "contract descriptor")
                search.require(isinstance(descriptor["path"], str)
                               and Path(descriptor["path"]).is_absolute()
                               and search.valid_hash(descriptor["sha256"]),
                               "prospective contract descriptor must use an absolute path and SHA256")
                contract = validate_contract(read_json(descriptor["path"], descriptor["sha256"]))
                checked = validate_fold_contract(contract, descriptor, plan, binding, objective, fold, resolved)
                search.require(checked["mode"] == AUTOSOMAL_PERSON_BATCHES,
                               "prospective K03 search requires AUTOSOMAL_PERSON_BATCHES_V1")
                descriptors[fold["id"]] = descriptor
            contexts[context_sha256] = {"request": expected_request, "folds": descriptors}
        for arm in search.CONTROLLED_ARMS:
            search.require(search.digest({"arm": arm, "contexts": contexts}) == spec["arm_input_sha256"][arm],
                           "arm input/context binding mismatch")
        search.require(plan["provenance"]["dataset_sha256"] ==
                       search.digest({"contexts": contexts, "scope": binding["scope"]}),
                       "dataset/context binding mismatch")
    else:
        search.require(binding["schema"] == BINDING_V1
                       and spec["primary_metric"] in ("component_mse", "component_log_loss")
                       and spec["evaluation_unit"] == "component_macro" and not spec["secondary_metrics"],
                       "this bounded trainer has no dense LAI/border adapter; do not substitute its selection metric")
        contracts = {}
        for fold in spec["folds"]:
            descriptor = binding["folds"][fold["id"]]
            contract = validate_contract(read_json(descriptor["path"], descriptor["sha256"]))
            validate_fold_contract(contract, descriptor, plan, binding, objective, fold)
            contracts[fold["id"]] = descriptor
        for arm in search.CONTROLLED_ARMS:
            search.require(search.digest({"arm": arm, "folds": contracts}) == spec["arm_input_sha256"][arm],
                           "arm input binding mismatch")
        search.require(plan["provenance"]["dataset_sha256"] ==
                       search.digest({"folds": contracts, "scope": binding["scope"]}),
                       "dataset/scope binding mismatch")
        contexts = {"legacy": {"request": None, "folds": contracts}}
    search.require(plan["provenance"]["trainer_code_sha256"] == sha(Path(__file__).with_name("r02_context_train.py")), "trainer source drift")
    search.require(binding["runtime_sha256"] == sha(Path(__file__).with_name("genomic_context_runtime.py")), "runtime source drift")
    return plan, binding, contexts


def prepare(plan_path, binding_path, binding_sha256, objective, state_dir, output, trial_number):
    plan, binding, contexts = validate_inputs(plan_path, binding_path, binding_sha256, objective)
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
    resolved_sha256 = sha(output / "resolved.json")
    search.require(search.digest(recipe["config"]) == recipe["tasks"][0]["config_sha256"],
                   "resolved configuration differs from committed recipe")
    context_sha256 = recipe["tasks"][0].get("request_context_sha256", "legacy")
    search.require(all(task.get("request_context_sha256", "legacy") == context_sha256
                       for task in recipe["tasks"]), "recipe mixes request contexts")
    search.require(context_sha256 in contexts, "recipe request context was not prospectively bound")
    contracts = contexts[context_sha256]["folds"]
    tasks = []
    for index, task in enumerate(recipe["tasks"]):
        record = {"recipe": recipe, "task": task, "contract": contracts[task["fold"]],
                  "scope": binding["scope"], "runtime_sha256": binding["runtime_sha256"],
                  "resolved_sha256": resolved_sha256,
                  "request_context": contexts[context_sha256]["request"]}
        path = output / f"task-{index:04d}.json"
        write_json(path, record)
        tasks.append({"path": str(path.resolve()), "sha256": sha(path)})
    write_json(output / "tasks.json", tasks)
    return tasks


def execute(task_path, task_sha256, resolved_path, output, checkpoint_root=None,
            checkpoint_every_steps=1, adam_state_precision="auto"):
    record = read_json(task_path, task_sha256)
    task = record["task"]
    resolved = read_json(resolved_path, record["resolved_sha256"])
    search.require(search.digest(resolved) == task["config_sha256"],
                   "resolved configuration content differs from task")
    if "request_context_sha256" in task:
        search.require(search.request_context(resolved) == record["request_context"]
                       and search.digest(record["request_context"]) == task["request_context_sha256"],
                       "resolved configuration/request context differs")
        expected_precision = resolved.get("adam_state_precision")
        search.require(expected_precision in ("fp32", "fp64"),
                       "prospective resolved configuration does not bind Adam state precision")
        if adam_state_precision == "auto":
            adam_state_precision = expected_precision
        search.require(adam_state_precision == expected_precision,
                       "requested Adam state precision differs from the prospective resolved configuration")
    elif adam_state_precision == "auto":
        # Historical plans predate an explicit precision field and used fp32.
        adam_state_precision = "fp32"
    search.require(record["recipe"]["provenance"]["trainer_code_sha256"] == sha(Path(__file__).with_name("r02_context_train.py")), "trainer source drift")
    search.require(record["runtime_sha256"] == sha(Path(__file__).with_name("genomic_context_runtime.py")), "runtime source drift")
    checkpoint_dir = ((Path(checkpoint_root) / task_sha256) if checkpoint_root is not None
                      else Path(output).parent / f".{Path(output).name}.{task_sha256}.checkpoints")
    receipt = train(record["contract"]["path"], record["contract"]["sha256"], resolved_path,
                    record["resolved_sha256"], task["arm"], task["seed"], output,
                    checkpoint_dir, checkpoint_every_steps, adam_state_precision)
    require_select_receipt(receipt)
    search.require(receipt["scope"] == record["scope"] and receipt["steps"] == task["checkpoint_step"], "training scope/budget drift")
    search.require(receipt["n_evaluated"] == task["n_evaluated"]
                   and receipt["evaluation_unit"] == task["evaluation_unit"], "training denominator drift")
    metric = record["recipe"]["primary_metric"]
    search.require(metric in receipt["metrics"], "requested metric not implemented by trainer")
    result = {**task, "status": "COMPLETE", "scope": "DEVELOPMENT", "metrics": {metric: receipt["metrics"][metric]},
              "checkpoint_sha256": receipt["checkpoint_sha256"], "prediction_sha256": receipt["prediction_sha256"]}
    if "request_context_sha256" in task:
        checkpoint = receipt["internal_checkpoint"]
        search.require(receipt["dataset_mode"] == task["dataset_mode"] == AUTOSOMAL_PERSON_BATCHES
                       and receipt["rare_pooling"] == resolved.get("rare_pooling", RARE_POOLING_SUM)
                       and receipt["locus_descriptor"] == resolved.get("locus_descriptor", LOCUS_DESCRIPTOR_LEGACY)
                       and checkpoint["schema"] == "r02_context_internal_checkpoint_v1"
                       and checkpoint["adam_state_precision"] == resolved["adam_state_precision"] == adam_state_precision
                       and checkpoint["final_confirmed_step"] == task["checkpoint_step"]
                       and 0 <= checkpoint["restored_steps"] <= checkpoint["final_confirmed_step"],
                       "training mode/checkpoint recovery identity differs")
        result.update(contract_sha256=receipt["contract_sha256"],
                      resolved_sha256=receipt["resolved_sha256"],
                      training_receipt_sha256=sha(Path(output) / "training.receipt.json"),
                      checkpoint_identity_sha256=checkpoint["identity_sha256"],
                      restored_steps=checkpoint["restored_steps"],
                      adam_state_precision=checkpoint["adam_state_precision"],
                      rare_pooling=receipt["rare_pooling"])
    write_json(Path(output) / "search-record.json", {"record": result, "actual_scope": receipt["scope"],
        "task_sha256": task_sha256, "training_receipt_sha256": sha(Path(output) / "training.receipt.json")})
    return result


def require_select_receipt(receipt):
    """FIT progress, even at the ceiling, is never a model-selection result.

    Old COMPLETE receipts predate the explicit role fields; their authenticated
    trainer always scored SELECT. New fields must agree when present.
    """
    search.require(receipt.get("status") == "COMPLETE"
                   and receipt.get("score_role", "SELECT") == "SELECT"
                   and receipt.get("select_accessed", True) is True,
                   "only completed SELECT training receipts can enter model selection")
    search.require(not receipt.get("dose_relocation"),
                   "post-selection dose relocation cannot enter primary recipe selection")


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
        require_select_receipt(training)
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
        if "request_context_sha256" in source["task"]:
            checkpoint = training["internal_checkpoint"]
            search.require(training["dataset_mode"] == source["task"]["dataset_mode"]
                           and training.get("locus_descriptor", LOCUS_DESCRIPTOR_LEGACY) ==
                           source["recipe"]["config"].get("locus_descriptor", LOCUS_DESCRIPTOR_LEGACY)
                           and result["record"]["contract_sha256"] == training["contract_sha256"]
                           and result["record"]["resolved_sha256"] == training["resolved_sha256"]
                           and result["record"]["training_receipt_sha256"] == result["training_receipt_sha256"]
                           and result["record"]["checkpoint_identity_sha256"] == checkpoint["identity_sha256"]
                           and result["record"]["restored_steps"] == checkpoint["restored_steps"]
                           and result["record"]["rare_pooling"] == training["rare_pooling"]
                           == source["recipe"]["config"].get("rare_pooling", RARE_POOLING_SUM)
                           and result["record"]["adam_state_precision"] == checkpoint["adam_state_precision"]
                           == source["recipe"]["config"]["adam_state_precision"],
                           "weighted person-batch receipt/recovery identity differs")
        search.require(sha(parent / "weights.npz") == training["checkpoint_sha256"], "missing/changed checkpoint")
        predictions = read_json(parent / "predictions.json", training["prediction_sha256"])
        for prediction in predictions:
            search.require(sha(prediction["path"]) == prediction["sha256"], "missing/changed prediction")
        search.require(result["record"]["metrics"] == training["metrics"], "metric receipt changed")
        seen.add(task_hash)
        records.append(result["record"])
        scopes.add(result["actual_scope"])
    search.require(seen == set(expected) and len(scopes) == 1, "incomplete/mixed-scope recipe")
    receipt_keys = ("schema", "scope", "objective", "trial_number", "recipe_sha256",
                    "plan_sha256", "provenance") + (("selection_design",)
                    if "selection_design" in recipe else ())
    receipt = {key: recipe[key] for key in receipt_keys}
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
    task_parser.add_argument("--adam-state-precision", choices=("auto", "fp32", "fp64"), default="auto")
    collect_parser = commands.add_parser("collect")
    for key in ("plan", "state-dir", "recipe", "task-index", "output"):
        collect_parser.add_argument("--" + key, required=True)
    collect_parser.add_argument("--results", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        plan, binding, contexts = validate_inputs(args.plan, args.binding, args.binding_sha256, args.objective)
        result = {"status": "AUTHENTICATED", "plan_sha256": search.digest(plan),
                  "binding_sha256": args.binding_sha256, "contexts": contexts}
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
