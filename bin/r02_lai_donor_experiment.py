#!/usr/bin/env python3
"""Run one recoverable FIT condition for autonomous donor-context LAI.

The JSON configuration fixes a prospective person-by-query plan.  Region
selection uses only person indices, query indices, the validity mask and a
seed recorded by the caller.  Truth is consulted later, solely to convert the
fixed query list into an executable loss mask and to report FIT-development
metrics.  Target/reference genotypes never select queries or regions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Mapping, Sequence
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from r02_lai_donor_context import AutonomousDonorContextLAI, STATE_NAMES
from r02_lai_donor_inputs import (
    CONDITIONS, DonorInputReader, RegionOwnership, region_plan_sha256,
    sources_from_spec,
)
from r02_lai_donor_train import (
    AdamSpecification, DONOR_OBJECTIVE, DonorTrainingBatch, FIT_SCOPE,
    LogicalBatchPlan, MicrobatchPlan, build_adam, donor_training_identity,
    run_donor_training_call, WEIGHTED_DONOR_OBJECTIVE, WeightedMicrobatchPlan,
)
from r02_lai_weighted_train import (
    WEIGHTED_REFERENCE_SCHEMA, load_weighted_fit_plan,
    validate_weighted_plan_reference,
)
from r02_context_checkpoint import identity_digest
from r02_local_attention_blocks import ContextExecution
from r02_lai_catalogue_policy import admit_catalogue_coverage, validate_catalogue_policy


SCHEMA = "r02_lai_donor_experiment_v2"
AVAILABLE_AXIS_SCHEMA = "r02_lai_donor_experiment_v3"
WEIGHTED_EXPERIMENT_SCHEMA = "r02_lai_donor_experiment_v4_occurrence_weighted"
UNUSED_REGION = "__UNUSED_NOT_ASSEMBLED__"
DOSAGE_BY_STATE = torch.tensor(
    ((2, 0, 0), (1, 1, 0), (1, 0, 1),
     (0, 2, 0), (0, 1, 1), (0, 0, 2)), dtype=torch.float64)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class QueryBlock:
    region_id: str
    person_index: int
    query_start: int
    query_stop: int

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "QueryBlock":
        require(isinstance(value, Mapping), "query block must be an object")
        require(set(value) == {"region_id", "person_index", "query_start", "query_stop"},
                "query block fields differ")
        block = cls(str(value["region_id"]), value["person_index"],
                    value["query_start"], value["query_stop"])
        require(block.region_id and block.region_id != UNUSED_REGION,
                "query block needs a nonreserved region_id")
        require(all(type(x) is int for x in
                    (block.person_index, block.query_start, block.query_stop))
                and block.person_index >= 0 and block.query_start >= 0
                and block.query_stop > block.query_start,
                "query block indices require a nonempty half-open interval")
        return block

    def receipt(self) -> dict[str, object]:
        return {"region_id": self.region_id, "person_index": self.person_index,
                "query_start": self.query_start, "query_stop": self.query_stop}


def parse_region_blocks(config: Mapping[str, object]) -> tuple[QueryBlock, ...]:
    plan = config.get("region_plan")
    require(isinstance(plan, Mapping) and set(plan) == {"method", "blocks"}
            and plan["method"] == "EXPLICIT_PERSON_CONTIGUOUS_QUERY_BLOCKS",
            "region_plan must explicitly declare contiguous person-query blocks")
    raw = plan["blocks"]
    require(isinstance(raw, list) and raw, "region_plan.blocks must be nonempty")
    blocks = tuple(QueryBlock.parse(value) for value in raw)
    require(len({block.region_id for block in blocks}) == len(blocks),
            "every physical person-query block requires a unique region_id")
    return blocks


def build_region_ownership(query_valid: np.ndarray,
                           blocks: Sequence[QueryBlock]) -> RegionOwnership:
    """Assign only declared blocks to loadable regions; all else stays unused."""
    valid = np.asarray(query_valid, dtype=np.bool_)
    require(valid.ndim == 2 and valid.size > 0, "query_valid must be a nonempty matrix")
    owner = np.where(valid, 0, -1).astype(np.int32)
    region_ids = (UNUSED_REGION,) + tuple(block.region_id for block in blocks)
    occupied = np.zeros(valid.shape, dtype=np.bool_)
    for number, block in enumerate(blocks, 1):
        require(block.person_index < valid.shape[0] and block.query_stop <= valid.shape[1],
                "region block is outside query axes")
        selected = np.zeros(valid.shape, dtype=np.bool_)
        selected[block.person_index, block.query_start:block.query_stop] = True
        selected &= valid
        require(bool(selected.any()), "region block has no valid query")
        require(not bool((occupied & selected).any()),
                "physical region blocks overlap on a valid person-query")
        owner[selected] = number
        occupied |= selected
    return RegionOwnership(region_ids, owner, region_plan_sha256(region_ids, owner))


def _referenced_blocks(config: Mapping[str, object],
                       blocks: Sequence[QueryBlock]) -> tuple[list[list[QueryBlock]], list[QueryBlock]]:
    by_id = {block.region_id: block for block in blocks}
    updates_raw = config.get("updates")
    require(isinstance(updates_raw, list) and updates_raw, "updates must be nonempty")
    updates: list[list[QueryBlock]] = []
    seen_global: set[tuple[int, int]] = set()
    for step, value in enumerate(updates_raw):
        require(isinstance(value, Mapping) and set(value) == {"update_id", "region_ids"},
                "update fields differ")
        require(value["update_id"] == f"u{step:03d}",
                "updates require ordered stable IDs u000, u001, ...")
        ids = value["region_ids"]
        require(isinstance(ids, list) and ids and len(ids) == len(set(ids)),
                "each update needs unique region_ids")
        require(all(region_id in by_id for region_id in ids),
                "update references an unknown region_id")
        chosen = [by_id[region_id] for region_id in ids]
        seen_update: set[tuple[int, int]] = set()
        for block in chosen:
            cells = {(block.person_index, index)
                     for index in range(block.query_start, block.query_stop)}
            require(seen_update.isdisjoint(cells),
                    "a person-query is duplicated within one update")
            require(seen_global.isdisjoint(cells),
                    "training is one pass: a person-query occurs in multiple updates")
            seen_update.update(cells); seen_global.update(cells)
        updates.append(chosen)
    report = config.get("fit_report")
    require(isinstance(report, Mapping)
            and set(report) == {"label", "region_ids"}
            and report["label"] == "FIT_DEVELOPMENT_FIXED_LIST",
            "fit_report must be an explicitly labelled fixed FIT list")
    report_ids = report["region_ids"]
    require(isinstance(report_ids, list) and report_ids
            and len(report_ids) == len(set(report_ids))
            and all(region_id in by_id for region_id in report_ids),
            "fit_report references unknown or duplicate region_ids")
    return updates, [by_id[region_id] for region_id in report_ids]


def _report_blocks(config: Mapping[str, object],
                   blocks: Sequence[QueryBlock]) -> list[QueryBlock]:
    """Resolve the unchanged fixed FIT report without parsing v1 updates."""
    by_id = {block.region_id: block for block in blocks}
    report = config.get("fit_report")
    require(isinstance(report, Mapping)
            and set(report) == {"label", "region_ids"}
            and report["label"] == "FIT_DEVELOPMENT_FIXED_LIST",
            "fit_report must be an explicitly labelled fixed FIT list")
    report_ids = report["region_ids"]
    require(isinstance(report_ids, list) and report_ids
            and len(report_ids) == len(set(report_ids))
            and all(region_id in by_id for region_id in report_ids),
            "fit_report references unknown or duplicate region_ids")
    return [by_id[region_id] for region_id in report_ids]


def _scored_indices(block: QueryBlock, query_valid: np.ndarray,
                    truth_observed: np.ndarray) -> tuple[int, ...]:
    # Block selection is already frozen.  This conversion only removes targets
    # for which no truth label exists; it never substitutes another query.
    keep = query_valid[block.person_index] & truth_observed[block.person_index]
    return tuple(index for index in range(block.query_start, block.query_stop)
                 if bool(keep[index]))


def build_logical_plans(updates: Sequence[Sequence[QueryBlock]], person_ids: Sequence[str],
                        query_valid: np.ndarray, truth_observed: np.ndarray
                        ) -> tuple[LogicalBatchPlan, ...]:
    require(query_valid.shape == truth_observed.shape,
            "query validity and truth observation axes differ")
    plans = []
    for step, blocks in enumerate(updates):
        ordered_people = tuple(dict.fromkeys(person_ids[b.person_index] for b in blocks))
        scored_by_person = {person: [] for person in ordered_people}
        microbatches = []
        for block in blocks:
            person = person_ids[block.person_index]
            scored = _scored_indices(block, query_valid, truth_observed)
            require(bool(scored), "fixed training block has no observed FIT target")
            scored_by_person[person].extend(scored)
            receipt = {"step_index": step, "block": block.receipt(),
                       "person_id": person, "scored_query_global_indices": list(scored)}
            microbatches.append(MicrobatchPlan(
                canonical_sha256(receipt), (person,), (scored,)))
        denominators = tuple(len(scored_by_person[person]) for person in ordered_people)
        plans.append(LogicalBatchPlan(
            step, f"u{step:03d}", ordered_people, denominators, tuple(microbatches)))
    return tuple(plans)


def _validate_model_config(config: Mapping[str, object]):
    model = config["model"]
    require(isinstance(model, Mapping), "model configuration required")
    expected = {"width", "heads", "depth", "feed_forward_multiplier",
                "locus_radius_cm", "query_radius_cm", "context_execution",
                "max_pair_state_elements", "max_query_state_elements"}
    optional = {"donor_block_size", "final_head_mode", "final_head_hidden_width"}
    require(expected <= set(model) <= expected | optional,
            "model configuration fields differ")
    head_fields = {"final_head_mode", "final_head_hidden_width"} & set(model)
    require(not head_fields or len(head_fields) == 2,
            "final head mode and hidden width must be specified together")
    if head_fields:
        mode, width = model["final_head_mode"], model["final_head_hidden_width"]
        require(mode in ("linear", "factorized_linear", "mlp"),
                "unsupported final head mode")
        require((mode == "linear" and width is None)
                or (mode != "linear" and type(width) is int and width >= len(STATE_NAMES)),
                "final head hidden width is incompatible with mode or output rank")
    execution = model["context_execution"]
    require(isinstance(execution, Mapping)
            and set(execution) == {"query_block_size", "max_mask_elements",
                                   "checkpoint_blocks", "local_mean_mode"},
            "context_execution fields differ")
    return model, execution


def _model(config: Mapping[str, object]) -> AutonomousDonorContextLAI:
    model, execution = _validate_model_config(config)
    return AutonomousDonorContextLAI(
        width=model["width"], heads=model["heads"], depth=model["depth"],
        feed_forward_multiplier=model["feed_forward_multiplier"],
        locus_radius_cm=model["locus_radius_cm"],
        query_radius_cm=model["query_radius_cm"],
        execution=ContextExecution(**execution),
        max_pair_state_elements=model["max_pair_state_elements"],
        max_query_state_elements=model["max_query_state_elements"],
        donor_block_size=model.get("donor_block_size"),
        final_head_mode=model.get("final_head_mode", "linear"),
        final_head_hidden_width=model.get("final_head_hidden_width"))


def _adam(config: Mapping[str, object]) -> AdamSpecification:
    value = config.get("optimizer")
    require(isinstance(value, Mapping), "optimizer configuration required")
    expected = {"learning_rate", "betas", "epsilon", "weight_decay",
                "gradient_clip_norm", "amsgrad", "maximize", "foreach",
                "capturable", "differentiable", "fused"}
    require(set(value) == expected, "optimizer configuration fields differ")
    copied = dict(value)
    require(isinstance(copied["betas"], list) and len(copied["betas"]) == 2,
            "optimizer betas must be a two-value JSON list")
    copied["betas"] = tuple(copied["betas"])
    return AdamSpecification(**copied)


def _assembled(reader: DonorInputReader, ownership: RegionOwnership,
               block: QueryBlock, config: Mapping[str, object], condition: str):
    model = config["model"]
    value = reader.assemble_region(
        ownership, block.region_id, condition=condition, depth=model["depth"],
        locus_radius_cm=model["locus_radius_cm"],
        query_radius_cm=model["query_radius_cm"],
        max_output_genotypes=config["reference_policy"]["max_output_genotypes"],
        person_indices=[block.person_index])
    policy = config["reference_policy"]
    require(value.lineage["reference_people"] == policy["expected_reference_people"],
            "reference count differs from the explicit policy")
    if policy["require_every_reference_allowed"]:
        require(sum(value.lineage["reference_counts_allowed_by_panel"].values()) ==
                value.lineage["reference_people"],
                "not every pinned reference is allowed for this source pool")
    return value


def _training_batch(reader: DonorInputReader, ownership: RegionOwnership,
                    block: QueryBlock, config: Mapping[str, object], condition: str,
                    microbatch: MicrobatchPlan | WeightedMicrobatchPlan) -> DonorTrainingBatch:
    value = _assembled(reader, ownership, block, config, condition)
    global_index = value.targets.query_global_index
    chosen = torch.zeros_like(value.batch.query_valid)
    for row, indices in enumerate(microbatch.scored_query_global_indices):
        for index in indices:
            chosen[row] |= global_index[row] == index
    chosen &= (value.batch.query_valid
               & (global_index >= block.query_start)
               & (global_index < block.query_stop))
    require(tuple(int(x) for x in global_index[0][chosen[0]
                    & value.targets.truth_observed[0]].tolist()) ==
            microbatch.scored_query_global_indices[0],
            "assembled scored queries differ from the frozen executable plan")
    return DonorTrainingBatch(
        microbatch.identity_sha256, microbatch.person_ids, value.batch,
        value.targets.truth_state, value.targets.truth_observed, chosen, global_index)


def empirical_prior(truth_state: np.ndarray, truth_observed: np.ndarray,
                    updates: Sequence[Sequence[QueryBlock]]) -> torch.Tensor:
    """Mean per-person six-state frequencies over the fixed training plan.

    A zero stays zero.  No unregistered pseudocount or clipping rule is added.
    """
    by_person: dict[int, list[QueryBlock]] = {}
    for update in updates:
        for block in update:
            by_person.setdefault(block.person_index, []).append(block)
    frequencies = []
    for person, blocks in sorted(by_person.items()):
        counts = np.zeros(len(STATE_NAMES), dtype=np.int64)
        for block in blocks:
            observed = truth_observed[person, block.query_start:block.query_stop]
            states = truth_state[person, block.query_start:block.query_stop][observed]
            counts += np.bincount(states.astype(np.int64), minlength=len(STATE_NAMES))
        require(int(counts.sum()) > 0,
                "a fixed training-plan person has no observed truth")
        frequencies.append(counts / counts.sum())
    require(bool(frequencies), "empirical FIT prior needs training-plan people")
    return torch.from_numpy(np.mean(frequencies, axis=0)).to(torch.float64)


def complete_fit_empirical_prior(truth_state: np.ndarray,
                                 truth_observed: np.ndarray,
                                 query_valid: np.ndarray) -> torch.Tensor:
    """Natural six-state mass of the complete FIT rectangle used by GT v1."""
    truth = np.asarray(truth_state)
    observed = np.asarray(truth_observed, dtype=np.bool_)
    valid = np.asarray(query_valid, dtype=np.bool_)
    require(truth.shape == observed.shape == valid.shape and truth.ndim == 2
            and bool(observed.all()) and bool(valid.all()),
            "weighted GT prior requires the complete FIT person-query rectangle")
    counts = np.bincount(truth.ravel().astype(np.int64), minlength=len(STATE_NAMES))
    require(len(counts) == len(STATE_NAMES) and int(counts.sum()) == truth.size,
            "weighted GT prior truth states differ")
    return torch.from_numpy(counts / counts.sum()).to(torch.float64)


def _mean_or_none(total: float, count: int, multiplier: int = 1):
    return total / (count * multiplier) if count else None


def evaluate(model: AutonomousDonorContextLAI, reader: DonorInputReader,
             ownership: RegionOwnership, blocks: Sequence[QueryBlock],
             config: Mapping[str, object], condition: str,
             prior: torch.Tensor) -> dict[str, object]:
    weighted_run = config["schema"] == WEIGHTED_EXPERIMENT_SCHEMA
    groups = {
        name: {"queries": 0, "nll": 0.0,
               "absolute_dosage_error": torch.zeros(3, dtype=torch.float64)}
        for name in ("all", "supported", "unsupported")
    }
    people: dict[int, dict[str, object]] = {}
    prior_nll = 0.0
    prior_nll_infinite = False
    prior_absolute_dosage_error = torch.zeros(3, dtype=torch.float64)
    true_dosage_table = DOSAGE_BY_STATE
    model.eval()
    with torch.no_grad():
        for block in blocks:
            value = _assembled(reader, ownership, block, config, condition)
            index = value.targets.query_global_index
            mask = (value.targets.truth_observed & value.batch.query_valid
                    & (index >= block.query_start) & (index < block.query_stop))
            require(bool(mask.any()), "fixed FIT report block has no observed target")
            output = model(value.batch)
            truth = value.targets.truth_state[mask]
            log_probability = F.log_softmax(output.logits[mask].double(), -1)
            probability = log_probability.exp()
            predicted_dosage = probability @ true_dosage_table
            true_dosage = true_dosage_table[truth]
            prior_dosage = prior @ true_dosage_table
            nll = -log_probability.gather(1, truth[:, None]).squeeze(1)
            dosage_error = (predicted_dosage - true_dosage).abs()
            supported = output.support[mask]
            for name, choose in (("all", torch.ones_like(supported)),
                                 ("supported", supported),
                                 ("unsupported", ~supported)):
                groups[name]["queries"] += int(choose.sum())
                groups[name]["nll"] += float(nll[choose].sum())
                groups[name]["absolute_dosage_error"] += dosage_error[choose].sum(0)
            person = people.setdefault(
                block.person_index,
                {"queries": 0, "nll": 0.0,
                 "absolute_dosage_error": torch.zeros(3, dtype=torch.float64)})
            person["queries"] += int(mask.sum())
            person["nll"] += float(nll.sum())
            person["absolute_dosage_error"] += dosage_error.sum(0)
            truth_prior = prior[truth]
            prior_nll_infinite |= bool((truth_prior == 0).any())
            if bool((truth_prior > 0).any()):
                prior_nll += float(-truth_prior[truth_prior > 0].log().sum())
            prior_absolute_dosage_error += (
                prior_dosage[None] - true_dosage).abs().sum(0)
    count = groups["all"]["queries"]
    metric_groups = {
        name: {
            "scored_queries": values["queries"],
            "mean_six_state_nll": _mean_or_none(values["nll"], values["queries"]),
            "mean_absolute_ancestry_dosage_error_per_query_and_ancestry":
                _mean_or_none(float(values["absolute_dosage_error"].sum()),
                              values["queries"], 3),
            "mean_absolute_ancestry_dosage_error_by_ancestry": {
                ancestry: _mean_or_none(float(values["absolute_dosage_error"][index]),
                                        values["queries"])
                for index, ancestry in enumerate(("AFR", "EUR", "NAM"))
            },
        }
        for name, values in groups.items()
    }
    person_equal = {
        "people": len(people),
        "mean_of_person_mean_six_state_nll": float(np.mean([
            value["nll"] / value["queries"] for value in people.values()])),
        "mean_of_person_mean_absolute_ancestry_dosage_error_by_ancestry": {
            ancestry: float(np.mean([
                value["absolute_dosage_error"][index].item() / value["queries"]
                for value in people.values()]))
            for index, ancestry in enumerate(("AFR", "EUR", "NAM"))
        },
    }
    return {
        "scope": "FIT_DEVELOPMENT_FIXED_LIST_SHARED_DONORS_NOT_INDEPENDENT_VALIDATION",
        "condition": condition,
        "experiment_configuration_sha256": canonical_sha256(config),
        "source_rare_coverage": dict(reader.catalogue_coverage),
        "catalogue_policy": config.get("catalogue_policy"),
        "fixed_list_sha256": canonical_sha256([block.receipt() for block in blocks]),
        "metrics": metric_groups,
        "person_equal_weighted_metrics": person_equal,
        "query_weighted_metrics_interpretation":
            "MICROCONDITIONAL_OVER_REPORTED_QUERIES_NOT_THE_TRAINING_OBJECTIVE_WHEN_PERSON_COUNTS_DIFFER",
        "support_fraction": groups["supported"]["queries"] / count,
        "empirical_six_state_fit_prior": dict(zip(STATE_NAMES, prior.tolist())),
        "empirical_prior_source":
            ("NATURAL_STATE_MASS_OVER_COMPLETE_FIT_PERSON_QUERY_RECTANGLE"
             if weighted_run else
             "MEAN_OF_PER_PERSON_STATE_FREQUENCIES_OVER_FIXED_TRAINING_PLAN"),
        "empirical_prior_mean_six_state_nll":
            None if prior_nll_infinite else prior_nll / count,
        "empirical_prior_nll_is_infinite": prior_nll_infinite,
        "empirical_prior_nll_null_reason":
            (("AT_LEAST_ONE_REPORTED_TRUTH_STATE_HAS_ZERO_COMPLETE_FIT_PRIOR_PROBABILITY"
              if weighted_run else
              "AT_LEAST_ONE_REPORTED_TRUTH_STATE_HAS_ZERO_TRAINING_PLAN_PRIOR_PROBABILITY")
             if prior_nll_infinite else None),
        "empirical_prior_mean_absolute_ancestry_dosage_error_per_query_and_ancestry":
            float(prior_absolute_dosage_error.sum()) / (count * 3),
        "empirical_prior_mean_absolute_ancestry_dosage_error_by_ancestry": {
            ancestry: float(prior_absolute_dosage_error[index]) / count
            for index, ancestry in enumerate(("AFR", "EUR", "NAM"))
        },
        "empirical_prior_limit":
            ("FREQUENCIES_USE_COMPLETE_FIT_RECTANGLE_AND_METRICS_USE_FIXED_FIT_REPORT_NO_INDEPENDENT_VALIDATION"
             if weighted_run else
             "FREQUENCIES_USE_FIXED_TRAINING_PLAN_AND_METRICS_USE_FIXED_FIT_REPORT_NO_INDEPENDENT_VALIDATION"),
    }


def load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    required = {"schema", "task_id", "plan_seed", "model_seed", "cpu_threads",
                "paired_conditions",
                "model", "optimizer", "reference_policy", "region_plan",
                "fit_report", "max_microbatches_per_call"}
    require(isinstance(value, dict), "experiment configuration must be an object")
    version = value.get("schema")
    required |= ({"catalogue_policy"} if version in
                 (AVAILABLE_AXIS_SCHEMA, WEIGHTED_EXPERIMENT_SCHEMA)
                 else {"require_full_catalogue_coverage"})
    required.add("weighted_fit_plan" if version == WEIGHTED_EXPERIMENT_SCHEMA
                 else "updates")
    require(set(value) == required and version in
            (SCHEMA, AVAILABLE_AXIS_SCHEMA, WEIGHTED_EXPERIMENT_SCHEMA),
            "experiment configuration schema or fields differ")
    require(isinstance(value["task_id"], str) and value["task_id"], "task_id required")
    require(type(value["plan_seed"]) is int and value["plan_seed"] >= 0,
            "nonnegative plan_seed required")
    require(type(value["model_seed"]) is int and value["model_seed"] >= 0,
            "nonnegative model_seed required")
    require(type(value["cpu_threads"]) is int and value["cpu_threads"] > 0,
            "positive cpu_threads required")
    require(value["paired_conditions"] == list(CONDITIONS),
            "paired_conditions must use ALL_AVAILABLE_SOURCE_RARE and NO_SOURCE_RARE in fixed order")
    if version == SCHEMA:
        require(value["require_full_catalogue_coverage"] is True,
                "legacy training requires complete source-rare catalogue coverage")
    else:
        validate_catalogue_policy(value["catalogue_policy"])
    policy = value["reference_policy"]
    require(isinstance(policy, dict)
            and set(policy) == {"expected_reference_people", "require_every_reference_allowed",
                                "max_output_genotypes"}
            and type(policy["expected_reference_people"]) is int
            and policy["expected_reference_people"] > 0
            and type(policy["require_every_reference_allowed"]) is bool
            and type(policy["max_output_genotypes"]) is int
            and policy["max_output_genotypes"] > 0,
            "explicit reference policy fields differ")
    maximum = value["max_microbatches_per_call"]
    require(maximum is None or (type(maximum) is int and maximum > 0),
            "max_microbatches_per_call must be positive or null")
    parse_region_blocks(value); _adam(value); _validate_model_config(value)
    if version == WEIGHTED_EXPERIMENT_SCHEMA:
        reference = value["weighted_fit_plan"]
        require(isinstance(reference, Mapping)
                and reference.get("schema") == WEIGHTED_REFERENCE_SCHEMA,
                "explicit weighted FIT plan reference required")
        validate_weighted_plan_reference(reference, path.parent)
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require_training_catalogue_coverage(sources, *, policy=None,
                                       input_spec_sha256=None) -> dict[str, object]:
    """Authenticate inputs and check the explicit, versioned locus scope."""
    with DonorInputReader(sources) as admission_reader:
        admission = dict(admission_reader.catalogue_coverage)
        # Use the classified COMMON+EVENT axis, not every row in the source
        # locus table (which may include sites absent from those channels).
        # A reader without an authenticated classified-axis hash fails closed.
        axis = admission.get("classified_material_axis_sha256")
    return admit_catalogue_coverage(admission, policy=policy,
        input_spec_sha256=input_spec_sha256, material_axis_sha256=axis)


def run(input_spec: Path, config_path: Path, condition: str,
        output_dir: Path) -> dict[str, object]:
    require(condition in CONDITIONS,
            "condition must be ALL_AVAILABLE_SOURCE_RARE or NO_SOURCE_RARE")
    config = load_config(config_path)
    sources = sources_from_spec(input_spec)
    # This admission occurs before any output directory, model, optimizer,
    # checkpoint, metric, prediction or inference object can be created.
    admission = require_training_catalogue_coverage(sources,
        policy=config.get("catalogue_policy"),
        input_spec_sha256=hashlib.sha256(input_spec.read_bytes()).hexdigest())
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    # The prospective plan was drawn outside this process with plan_seed and is
    # now explicit.  Runtime/model RNG has a separate namespace.
    random.seed(config["model_seed"]); np.random.seed(config["model_seed"])
    torch.manual_seed(config["model_seed"])
    model = _model(config)
    specification = _adam(config)
    optimizer = build_adam(model, specification)
    blocks = parse_region_blocks(config)
    weighted_run = config["schema"] == WEIGHTED_EXPERIMENT_SCHEMA
    if weighted_run:
        updates = None
        report_blocks = _report_blocks(config, blocks)
        weighted_reference = validate_weighted_plan_reference(
            config["weighted_fit_plan"], config_path.parent)
    else:
        updates, report_blocks = _referenced_blocks(config, blocks)
        weighted_reference = None
    with DonorInputReader(sources) as reader:
        query_valid = np.asarray(reader.query["query_valid"], dtype=np.bool_)
        truth_observed = np.asarray(reader.query["truth_observed"], dtype=np.bool_)
        truth_state = np.asarray(reader.query["truth_state"], dtype=np.int64)
        person_ids = tuple(reader.query["person_id"].astype(str).tolist())
        report_ownership = build_region_ownership(query_valid, report_blocks)
        weighted_conversion = None
        report_truth_state_counts = None
        if weighted_run:
            weighted_conversion = load_weighted_fit_plan(
                weighted_reference, person_ids=person_ids,
                query_valid=query_valid, truth_observed=truth_observed,
                truth_state=truth_state, source_spec_path=input_spec,
                query_truth_overlay_path=reader.paths["query_overlay"])
            require(config["plan_seed"] ==
                    weighted_conversion.plan_metadata.get("seed"),
                    "experiment plan_seed differs from frozen GT occurrence draw")
            training_blocks = tuple(QueryBlock(
                value.region_id, value.person_index,
                value.query_start, value.query_stop)
                for value in weighted_conversion.physical_segments)
            training_ownership = build_region_ownership(query_valid, training_blocks)
            plans = weighted_conversion.logical_batches
            prior = complete_fit_empirical_prior(
                truth_state, truth_observed, query_valid)
            report_truth_state_counts_array = np.zeros(len(STATE_NAMES), dtype=np.int64)
            for block in report_blocks:
                keep = (query_valid[block.person_index, block.query_start:block.query_stop]
                        & truth_observed[block.person_index,
                                         block.query_start:block.query_stop])
                states = truth_state[block.person_index,
                                     block.query_start:block.query_stop][keep]
                report_truth_state_counts_array += np.bincount(
                    states, minlength=len(STATE_NAMES))
            report_truth_state_counts = report_truth_state_counts_array.tolist()
        else:
            training_blocks = blocks
            training_ownership = build_region_ownership(query_valid, training_blocks)
            plans = build_logical_plans(updates, person_ids,
                                        query_valid, truth_observed)
            prior = empirical_prior(truth_state, truth_observed, updates)
        initial_path = output_dir / "FIT_METRICS_INITIAL.json"
        if initial_path.exists():
            initial = json.loads(initial_path.read_text())
            require(initial.get("fixed_list_sha256") ==
                    canonical_sha256([block.receipt() for block in report_blocks])
                    and initial.get("condition") == condition
                    and initial.get("experiment_configuration_sha256") ==
                        canonical_sha256(config),
                    "existing initial FIT metrics use a different condition, config, or fixed list")
        else:
            initial = evaluate(model, reader, report_ownership, report_blocks,
                               config, condition, prior)
            _atomic_json(initial_path, initial)

        code_root = Path(__file__).resolve().parent
        code_files = (
            "r02_lai_donor_experiment.py", "r02_lai_donor_inputs.py",
            "r02_lai_source_rare_catalogue.py",
            "r02_lai_catalogue_policy.py",
            "r02_lai_donor_train.py", "r02_lai_donor_context.py",
            "r02_lai_common_context.py", "r02_context_checkpoint.py",
            "r02_context_update_checkpoint.py", "r02_local_attention_blocks.py",
            "r02_lai_individual_reference.py", "r02_lai_donor_eligibility.py",
            "r02_lai_role_units.py",
        )
        source_paths = {"input_spec": input_spec, "experiment_config": config_path}
        source_paths.update({f"code:{name}": code_root / name for name in code_files})
        if weighted_run:
            source_paths.update({
                "weighted_plan_json": weighted_reference.plan_json,
                "weighted_occurrences_npz": weighted_reference.occurrences_npz,
                "code:r02_lai_weighted_train.py": code_root / "r02_lai_weighted_train.py",
            })
        identity = donor_training_identity(
            task_id=f'{config["task_id"]}:{condition}', data_scope=FIT_SCOPE,
            model=model, optimizer=optimizer, optimizer_specification=specification,
            source_paths=source_paths,
            policy_path=config_path,
            objective=(WEIGHTED_DONOR_OBJECTIVE if weighted_run else DONOR_OBJECTIVE),
            seed=config["model_seed"],
            logical_batches=plans)
        identity["source_rare_coverage"] = admission
        if weighted_run:
            block_by_id = {block.region_id: block for block in training_blocks}
        else:
            flat = [block for update in updates for block in update]
            offsets = []
            cursor = 0
            for update in updates:
                offsets.append(cursor); cursor += len(update)

        def batch_source(_model, step, index):
            microbatch = plans[step].microbatches[index]
            block = (block_by_id[microbatch.region_id] if weighted_run
                     else flat[offsets[step] + index])
            return _training_batch(reader, training_ownership, block, config, condition,
                                   microbatch)

        progress_path = output_dir / "PROGRESS.json"
        prior_progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        elapsed_before = float(prior_progress.get("elapsed_seconds", 0.0))
        invocation_started = time.monotonic()
        started_at = prior_progress.get("started_at") or datetime.now(timezone.utc).isoformat()
        model.train()
        while True:
            result = run_donor_training_call(
                model, optimizer, specification, output_dir / "checkpoints",
                identity, plans, batch_source,
                max_microbatches_this_call=config["max_microbatches_per_call"])
            _atomic_json(progress_path, {
                "schema": ("r02_lai_donor_experiment_progress_v2_occurrence_weighted"
                           if weighted_run else "r02_lai_donor_experiment_progress_v1"),
                "status": ("CHECKPOINT_SEQUENCE_COMPLETE" if
                           result.completed_steps == len(plans) else "RUNNING"),
                "condition": condition, "started_at": started_at,
                "source_rare_coverage": admission,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed_before + time.monotonic() - invocation_started,
                "completed_steps": result.completed_steps,
                "total_steps": len(plans), "logical_batch_id": result.logical_batch_id,
                "next_microbatch": result.next_microbatch,
                ("history_weighted_occurrence_nll" if weighted_run
                 else "history_mean_person_nll"): list(result.history),
                "last_call_unsupported_scored_queries": result.unsupported_scored_queries,
                **({"last_call_unsupported_occurrences":
                    result.unsupported_occurrences} if weighted_run else {}),
            })
            # COMPLETE means this call durably finished one logical update;
            # it does not mean the whole declared sequence is complete.
            if result.completed_steps == len(plans):
                break
        final = evaluate(model, reader, report_ownership, report_blocks,
                         config, condition, prior)
        _atomic_json(output_dir / "FIT_METRICS_FINAL.json", final)

    receipt = {
        "schema": ("r02_lai_donor_experiment_result_v2_occurrence_weighted"
                   if weighted_run else "r02_lai_donor_experiment_result_v1"),
        "status": result.status,
        "task_id": config["task_id"], "condition": condition,
        "plan_seed": config["plan_seed"], "model_seed": config["model_seed"],
        "training_identity_sha256": identity_digest(identity),
        "condition_is_not_model_initialization_input": True,
        "completed_updates": result.completed_steps,
        ("history_weighted_occurrence_nll" if weighted_run
         else "history_mean_person_nll"): list(result.history),
        "unsupported_scored_queries_last_call": result.unsupported_scored_queries,
        **({"unsupported_occurrences_last_call": result.unsupported_occurrences}
           if weighted_run else {}),
        "initial_fit_metrics": initial, "final_fit_metrics": final,
        "scientific_scope": "FIT_DEVELOPMENT_SHARED_DONORS_NO_SELECT_OR_EVALUATE",
        "source_rare_coverage": admission,
        "catalogue_coverage_scope":
            "NECESSARY_AXIS_MEMBERSHIP_ONLY_NOT_REFERENCE_CALLABILITY_OR_SCIENTIFIC_ADMISSION",
    }
    if weighted_run:
        receipt["weighted_fit_training_conversion"] = weighted_conversion.source_receipt
        receipt["training_objective"] = WEIGHTED_DONOR_OBJECTIVE
        receipt["fixed_fit_report_truth_state_counts"] = dict(
            zip(STATE_NAMES, report_truth_state_counts))
        receipt["fixed_fit_report_has_all_six_states"] = all(
            value > 0 for value in report_truth_state_counts)
        receipt["fixed_fit_report_scope"] = (
            "UNCHANGED_FIXED_FIT_DEVELOPMENT_LIST;WEIGHTED_SIX_STATE_TRAINING_DOES_NOT_"
            "MAKE_THIS_REPORT_COMPLETE_OR_INDEPENDENT")
    _atomic_json(output_dir / "RESULT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_spec, args.config, args.condition, args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
