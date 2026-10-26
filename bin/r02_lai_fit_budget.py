#!/usr/bin/env python3
"""Measure one authenticated LAI FIT update without opening SELECT/EVALUATE.

This is a resource measurement, not a training campaign.  ``inventory`` ranks
the 16 chr22 regions from predictor geometry only.  ``measure`` reconstructs
one already complete FIT/NONE factorized link, executes forward, the existing
six-state cross-entropy, backward and Adam once, then proves that the resulting
technical checkpoint restores exactly.  The checkpoint is permanently marked
``EXCLUDE_FROM_SCIENTIFIC_SELECTION``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r02_context_train as trainer
from r02_context_checkpoint import CheckpointStore
from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE


INVENTORY_SCHEMA = "r02_lai_fit_budget_inventory_v1"
PRE_SCHEMA = "r02_lai_fit_budget_measurement_pre_v1"
RECEIPT_SCHEMA = "r02_lai_fit_budget_measurement_v1"
EXCLUSION = "EXCLUDE_FROM_SCIENTIFIC_SELECTION"
FAMILIES = ("lai_cnn", "lai_local_attention")
DEFAULT_THREADS = 1
DEFAULT_MEMORY_BYTES = 8 * 1024**3
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_WIDTH = 64
DEFAULT_RADIUS_CM = 1.0
DEFAULT_CONDITION = "NONE"
DEFAULT_SEED = 1701
LEARNING_RATE_LOW = 1e-4
LEARNING_RATE_HIGH = 1e-3
MEASUREMENT_LEARNING_RATE = LEARNING_RATE_LOW
CHECKPOINT_RECOVERY = "POST_ADAM_MODEL_OPTIMIZER_AND_RANDOM_STATES"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def descriptor(path, expected_sha=None):
    path = Path(path).resolve()
    require(path.is_file(), "source file is absent: " + str(path))
    observed = sha256(path)
    require(expected_sha is None or observed == expected_sha,
            "source SHA-256 differs: " + str(path))
    return {"path": str(path), "sha256": observed}


def read_json(spec):
    path = descriptor(spec["path"], spec["sha256"])["path"]
    return json.loads(Path(path).read_text())


def write_fresh(path, value):
    path = Path(path)
    require(not path.exists() and path.parent.is_dir(),
            "fresh output file required: " + str(path))
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               allow_nan=False) + "\n")
    return descriptor(path)


def _npy_shape(archive, name):
    value = archive[name]
    require(value.ndim == 2 and value.shape[0] == 128,
            name + " must contain the retained 128 mosaics")
    return tuple(map(int, value.shape))


def geometry_counts(batch_path):
    """Read masks and shapes only; no simulated ancestry state is opened."""
    with np.load(batch_path, allow_pickle=False) as archive:
        required = {"person_id", "common_valid", "event_valid", "query_valid"}
        require(required <= set(archive.files), "predictor geometry arrays are incomplete")
        require(archive["person_id"].shape == (128,),
                "predictor does not retain exactly 128 mosaics")
        result = {"people": 128}
        for prefix in ("common", "event", "query"):
            mask = archive[prefix + "_valid"]
            shape = _npy_shape(archive, prefix + "_valid")
            require(mask.dtype == bool, prefix + " validity mask is not boolean")
            result[prefix + "_sites"] = shape[1]
            result[prefix + "_valid_values"] = int(mask.sum())
        result["compressed_bytes"] = int(Path(batch_path).stat().st_size)
    return result


def cost_proxy(family, geometry, width=DEFAULT_WIDTH, depth=2,
               cnn_kernel_sizes=(3, 101)):
    """Deterministic ranking proxy; it is not a runtime or RAM estimate."""
    people = geometry["people"]
    common = geometry["common_sites"]
    event = geometry["event_sites"]
    query = geometry["query_sites"]
    if family == "lai_cnn":
        work = people * width * (common + event + depth * query * sum(cnn_kernel_sizes))
    elif family == "lai_local_attention":
        work = people * width * (common + query + event + depth * event * event)
    else:
        raise ValueError("unknown LAI family")
    return int(work)


def locate_request_by_sha(work_root, expected_sha):
    matches = []
    for path in Path(work_root).glob("*/*/link-request.json"):
        if sha256(path) == expected_sha:
            matches.append(path.resolve())
    require(len(matches) == 1,
            "exactly one retained Nextflow link request must match the receipt SHA-256")
    return descriptor(matches[0], expected_sha)


def _verify_predictor(request_spec, prepared_root, predictor_id, family,
                      task_id, radius_cm):
    request = read_json(request_spec)
    require(request.get("schema") == "r02_lai_regional_predictor_request_v1"
            and request.get("operation") == "PREPARE"
            and request.get("predictor_id", predictor_id) == predictor_id
            and request.get("task_id") == task_id
            and request.get("role") == "FIT"
            and str(request.get("chromosome")) == "22"
            and request.get("family") == family
            and float(request.get("scale_cm")) == radius_cm
            and request.get("training_allowed") is False
            and request.get("selection_performed") is False
            and request.get("evaluate_opened") is False,
            "predictor request differs from the FIT chr22 PREPARE source")
    prepared = Path(prepared_root) / "predictors" / predictor_id / "prepared"
    receipt_spec = descriptor(prepared / "receipt.json")
    batch_spec = descriptor(prepared / "regional-source.npz")
    receipt = read_json(receipt_spec)
    require(receipt.get("schema") == "r02_lai_regional_predictor_receipt_v1"
            and receipt.get("status") ==
                "REGIONAL_PREDICTORS_PREPARED_NO_TARGETS_CONTROLS_TRAINING_SELECTION_OR_EVALUATION"
            and receipt.get("request_sha256") == request_spec["sha256"]
            and receipt.get("batch_sha256") == batch_spec["sha256"]
            and receipt.get("task_id") == task_id and receipt.get("role") == "FIT"
            and receipt.get("family") == family
            and float(receipt.get("scale_cm")) == radius_cm
            and receipt.get("training_allowed") is False
            and receipt.get("truth_loaded") is False
            and receipt.get("selection_performed") is False
            and receipt.get("evaluate_opened") is False,
            "prepared predictor receipt is incomplete or has opened a forbidden role")
    return request, receipt_spec, batch_spec, geometry_counts(batch_spec["path"])


def build_inventory(plan_path, plan_sha256, prepared_root, task_id, output,
                    radius_cm=DEFAULT_RADIUS_CM, width=DEFAULT_WIDTH):
    plan_spec = descriptor(plan_path, plan_sha256)
    plan = read_json(plan_spec)
    require(plan.get("schema") == "r02_lai_factorized_regional_request_plan_v1"
            and plan.get("training_authorized") is False
            and plan.get("selection_authorized") is False
            and plan.get("evaluate_opened") is False,
            "factorized PREPARE plan differs or opens training/selection/evaluation")
    by_id = {row["predictor_id"]: row for row in plan.get("predictors", [])}
    require(len(by_id) == len(plan.get("predictors", [])), "duplicate predictor ID in plan")
    selections = {}
    work_root = Path(prepared_root).parent / "work"
    require(work_root.is_dir(), "retained Nextflow work directory is required")
    for family in FAMILIES:
        candidates = []
        for predictor_id, row in by_id.items():
            request_spec = row["request"]
            request = read_json(request_spec)
            if not (request.get("task_id") == task_id and request.get("family") == family
                    and float(request.get("scale_cm", -1)) == radius_cm):
                continue
            request, receipt_spec, batch_spec, geometry = _verify_predictor(
                request_spec, prepared_root, predictor_id, family, task_id, radius_cm)
            kernels = request["expected_layout_recipe"].get(
                "cnn_query_config", {}).get("kernel_sizes", [3, 101])
            proxy = cost_proxy(family, geometry, width, request["expected_layout_recipe"]["depth"],
                               tuple(kernels))
            link_id = predictor_id + "." + DEFAULT_CONDITION
            template_row = next((item for item in plan["links"]
                                 if item["link_id"] == link_id), None)
            require(template_row is not None, "NONE link template absent for predictor")
            link_receipt_path = (Path(prepared_root) / "links" / link_id /
                                 "linked" / "receipt.json")
            candidates.append({
                "predictor_id": predictor_id, "region_id": request["region_id"],
                "predictor_request": request_spec,
                "predictor_receipt": receipt_spec, "predictor_batch": batch_spec,
                "link_template": template_row["template"],
                "link_receipt_path": str(link_receipt_path.resolve()),
                "link_receipt_complete": link_receipt_path.is_file(),
                "geometry": geometry, "cost_proxy": proxy,
            })
        require(len(candidates) == 16,
                f"{family} inventory must contain the 16 complete predictor regions")
        candidates.sort(key=lambda row: (row["cost_proxy"], row["region_id"]))
        selected = candidates[-1]
        link_receipt_spec = descriptor(selected["link_receipt_path"])
        link_receipt = read_json(link_receipt_spec)
        link_request_spec = locate_request_by_sha(work_root, link_receipt["request_sha256"])
        link_request = read_json(link_request_spec)
        require(link_request.get("schema") == "r02_lai_regional_truth_join_request_v2"
                and link_request.get("operation") == "PREPARE"
                and link_request.get("role") == "FIT"
                and link_request.get("condition") == DEFAULT_CONDITION
                and link_request.get("training_authorized") is False
                and link_request.get("selection_authorized") is False
                and link_request.get("evaluate_opened") is False
                and link_request.get("regional_request", {}).get("sha256") ==
                    selected["predictor_request"]["sha256"]
                and link_request.get("regional_receipt", {}).get("sha256") ==
                    selected["predictor_receipt"]["sha256"]
                and link_request.get("regional_batch", {}).get("sha256") ==
                    selected["predictor_batch"]["sha256"],
                "retained link request does not bind the published predictor copies")
        # Open every descriptor now so a dangling Nextflow work link is not
        # mistaken for a complete source.
        for name in ("regional_request", "regional_receipt", "regional_batch",
                     "overlay", "overlay_receipt", "control_request", "control_receipt",
                     "control_batch", "event_burden_prefix", "event_burden_prefix_receipt"):
            descriptor(link_request[name]["path"], link_request[name]["sha256"])
        selected["link_request"] = link_request_spec
        selected["link_receipt"] = link_receipt_spec
        values = [row["cost_proxy"] for row in candidates]
        selections[family] = {
            "selection_rule":
                "MAX_ARCHITECTURE_AWARE_GEOMETRY_PROXY_BEFORE_OPENING_TRUTH",
            "selection_reason":
                "Stress the retained 128-mosaic region with the largest family-specific proxy; "
                "this is an upper-cost probe, not the typical region or a biological choice.",
            "proxy_formula": ("people*width*(common_sites+event_sites+depth*query_sites*sum(kernels))"
                              if family == "lai_cnn" else
                              "people*width*(common_sites+query_sites+event_sites+depth*event_sites^2)"),
            "median_proxy": statistics.median(values), "maximum_proxy": max(values),
            "selected": selected, "candidates": candidates,
        }
    result = {
        "schema": INVENTORY_SCHEMA,
        "status": "SOURCES_RANKED_WITHOUT_OPENING_SIMULATED_ANCESTRY_TRUTH",
        "plan": plan_spec, "prepared_root": str(Path(prepared_root).resolve()),
        "retained_nextflow_work_root": str(work_root.resolve()),
        "task_id": task_id, "chromosome": "22", "role": "FIT",
        "condition": DEFAULT_CONDITION, "people_per_region": 128,
        "families": list(FAMILIES), "radius_cm": radius_cm, "width": width,
        "threads": DEFAULT_THREADS, "memory_bytes": DEFAULT_MEMORY_BYTES,
        "timeout_seconds_per_family": DEFAULT_TIMEOUT_SECONDS,
        "selection_opened": False, "evaluate_opened": False,
        "truth_opened_for_region_choice": False,
        "complete_campaign_required_for_measurement": False,
        "selections": selections,
        "scientific_use": EXCLUSION,
        "limitations": [
            "The proxy ranks these 16 chr22 regions; it does not predict seconds or RAM.",
            "The selected maximum is deliberately not a median or a uniform-genome claim.",
            "One update cannot determine convergence, stopping, total budget or model selection.",
        ],
    }
    write_fresh(output, result)
    return result


def validate_pre(pre, inventory_spec, family):
    required = {"schema", "status", "scope", "inventory", "family", "chromosome",
                "role", "condition", "radius_cm", "width", "updates", "threads",
                "memory_bytes", "timeout_seconds", "seed", "learning_rate_domain",
                "measurement_learning_rate", "checkpoint_recovery", "selection_opened",
                "evaluate_opened", "scientific_use"}
    require(isinstance(pre, dict) and set(pre) == required
            and pre["schema"] == PRE_SCHEMA
            and pre["status"] == "ADMIT_ONE_TECHNICAL_FIT_UPDATE"
            and pre["scope"] == "DEVELOPMENT_BUDGET_MEASUREMENT"
            and pre["inventory"] == inventory_spec
            and pre["family"] == family in FAMILIES
            and pre["chromosome"] == "22" and pre["role"] == "FIT"
            and pre["condition"] == DEFAULT_CONDITION
            and pre["radius_cm"] == DEFAULT_RADIUS_CM
            and pre["width"] == DEFAULT_WIDTH and pre["updates"] == 1
            and pre["threads"] == DEFAULT_THREADS
            and pre["memory_bytes"] == DEFAULT_MEMORY_BYTES
            and pre["timeout_seconds"] == DEFAULT_TIMEOUT_SECONDS
            and pre["seed"] == DEFAULT_SEED
            and pre["learning_rate_domain"] == {
                "low": LEARNING_RATE_LOW, "high": LEARNING_RATE_HIGH,
                "spacing": "GEOMETRIC"}
            and math.isclose(pre["measurement_learning_rate"],
                             MEASUREMENT_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-18)
            and pre["checkpoint_recovery"] == CHECKPOINT_RECOVERY
            and pre["selection_opened"] is False and pre["evaluate_opened"] is False
            and pre["scientific_use"] == EXCLUSION,
            "PRE does not admit this exact one-update technical measurement")


def resolved_configuration(family, predictor_request, width=DEFAULT_WIDTH):
    recipe = predictor_request["expected_layout_recipe"]
    resolved = {
        "family": family, "common_radius_cm": float(recipe["common_radius_cm"]),
        "rare_radius_cm": float(recipe["rare_radius_cm"]), "width": width,
        "depth": int(recipe["depth"]), "dropout": 0.0,
        "heads": 4 if family == "lai_local_attention" else None,
        # This value is a measurement input from the frozen candidate domain;
        # the probe neither selects nor recommends it.
        "learning_rate": MEASUREMENT_LEARNING_RATE, "optimizer": "Adam",
        "learning_rate_measurement_rule":
            "LOW_ENDPOINT_OF_FROZEN_GEOMETRIC_DOMAIN_FOR_COST_MEASUREMENT_NOT_SELECTION",
    }
    if family == "lai_cnn":
        resolved["query_max_gap_cm"] = float(recipe["query_max_gap_cm"])
        resolved["cnn_query_config"] = recipe["cnn_query_config"]
    return resolved


def measurement_contract(data, max_batch_bytes=DEFAULT_MEMORY_BYTES):
    persons = data["person_id"].astype(str).tolist()
    components = data["component_id"].astype(str).tolist()
    reference_identity = {
        "measurement_source_arrays_sha256": hashlib.sha256(
            canonical(sorted(data))).hexdigest()
    }
    return {
        "objective": "lai",
        "persons": {person: {"role": "FIT", "component": component}
                    for person, component in zip(persons, components)},
        "limits": {"max_common_sites": int(data["common_valid"].shape[1]),
                   "max_event_sites": int(data["event_valid"].shape[1]),
                   "max_targets": int(data["query_valid"].shape[1]),
                   "max_batch_bytes": max_batch_bytes},
        "common_features": int(data["common_features"].shape[2]),
        "event_metadata_features": int(data["event_metadata"].shape[2]),
        "event_reference_features": 15,
        "event_reference_feature_schema": "r02_lai_event_reference_features_v1",
        "event_reference_ancestry_order": ["African", "European", "Native_American"],
        "reference_identity": reference_identity,
        "lai_burden_scope": SCOPE, "lai_burden_boundary": BOUNDARY,
        "lai_burden_count_names": list(COUNT_NAMES),
        "lai_burden_transform": "LOG1P_IN_MODEL",
        "query_max_gap_cm": None,
    }


def tensor_state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(canonical(list(array.shape))); digest.update(array.tobytes())
    return digest.hexdigest()


def state_digest(value):
    """Hash nested optimizer or random state without pickle serialization."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor"); digest.update(str(array.dtype).encode())
            digest.update(canonical(list(array.shape))); digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray"); digest.update(str(array.dtype).encode())
            digest.update(canonical(list(array.shape))); digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: canonical(str(candidate))):
                visit(key); visit(item[key])
        elif isinstance(item, tuple):
            digest.update(b"tuple")
            for child in item: visit(child)
        elif isinstance(item, list):
            digest.update(b"list")
            for child in item: visit(child)
        else:
            digest.update(b"scalar"); digest.update(canonical(item))

    visit(value)
    return digest.hexdigest()


def peak_rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def enforced_process_limits(memory_ceiling=DEFAULT_MEMORY_BYTES, cpu_ceiling=1.0):
    """Require an external cgroup bound; declarations inside JSON are insufficient."""
    memory_path = Path("/sys/fs/cgroup/memory.max")
    cpu_path = Path("/sys/fs/cgroup/cpu.max")
    require(memory_path.is_file() and cpu_path.is_file(),
            "cgroup-v2 memory.max and cpu.max are required for human measurement")
    memory_text = memory_path.read_text().strip()
    require(memory_text.isdigit(), "human measurement requires a finite cgroup memory.max")
    memory_max = int(memory_text)
    require(0 < memory_max <= memory_ceiling,
            "cgroup memory.max exceeds the admitted 8 GiB ceiling")
    cpu_fields = cpu_path.read_text().split()
    require(len(cpu_fields) == 2 and cpu_fields[0].isdigit() and cpu_fields[1].isdigit(),
            "human measurement requires a finite cgroup cpu.max")
    quota, period = map(int, cpu_fields)
    cpus = quota / period
    require(0 < cpus <= cpu_ceiling,
            "cgroup cpu.max exceeds the admitted one-CPU ceiling")
    return {"mechanism": "CGROUP_V2", "memory_max_bytes": memory_max,
            "cpu_quota": quota, "cpu_period": period, "cpu_ceiling": cpus}


class Deadline:
    def __init__(self, seconds): self.seconds = seconds; self.previous = None
    def __enter__(self):
        require(self.seconds == DEFAULT_TIMEOUT_SECONDS,
                "measurement timeout must remain 600 seconds per family")
        if hasattr(signal, "SIGALRM"):
            self.previous = signal.signal(signal.SIGALRM,
                lambda *_: (_ for _ in ()).throw(TimeoutError("measurement exceeded 600 seconds")))
            signal.alarm(self.seconds)
        return self
    def __exit__(self, *_):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0); signal.signal(signal.SIGALRM, self.previous)


def run_update(data, resolved, output, *, threads=DEFAULT_THREADS,
               memory_bytes=DEFAULT_MEMORY_BYTES, seed=DEFAULT_SEED,
               source_identity=None):
    """Run and preserve one technical update; callable with synthetic fixtures."""
    output = Path(output)
    require(not output.exists() and output.parent.is_dir(), "fresh measurement output required")
    require(threads == 1 and memory_bytes == DEFAULT_MEMORY_BYTES,
            "measurement resources must remain 1 CPU thread and 8 GiB")
    require(data["person_id"].shape == (128,) or
            (source_identity or {}).get("synthetic_code_test") is True,
            "human measurement must retain exactly 128 mosaics")
    require("query_core" in data and data["query_core"].dtype == bool
            and data["query_core"].shape == data["query_valid"].shape,
            "measurement requires the authenticated query_core mask")
    valid_core_queries = data["query_core"] & data["query_valid"]
    core_objectives = valid_core_queries & data["truth_observed"]
    require(bool(core_objectives.any(axis=1).all()),
            "measurement requires at least one observed core objective per mosaic")
    output.mkdir(mode=0o700)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(threads)
    contract = measurement_contract(data, memory_bytes)
    contract["query_max_gap_cm"] = resolved.get("query_max_gap_cm")
    trainer.validate_batch(data, contract, "FIT", action="TRAIN")
    phases = {}
    started = time.perf_counter()
    model = trainer.TaskModel(resolved, contract)
    optimizer = trainer.build_optimizer(model.parameters(),
                                        model.encoder.config.learning_rate, "fp32")
    phases["initialize_model_optimizer_seconds"] = time.perf_counter() - started
    model.train(); optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter(); prediction = model(data, DEFAULT_CONDITION)
    phases["forward_seconds"] = time.perf_counter() - started
    started = time.perf_counter(); loss = trainer.training_loss(prediction, data, "lai")
    require(bool(torch.isfinite(loss)), "nonfinite technical measurement loss")
    phases["loss_seconds"] = time.perf_counter() - started
    started = time.perf_counter(); loss.backward()
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in model.parameters()), "nonfinite technical measurement gradient")
    phases["backward_seconds"] = time.perf_counter() - started
    started = time.perf_counter(); optimizer.step()
    phases["adam_seconds"] = time.perf_counter() - started
    expected_optimizer_digest = state_digest(optimizer.state_dict())
    expected_python_rng_digest = state_digest(random.getstate())
    expected_numpy_rng_digest = state_digest(np.random.get_state())
    expected_torch_rng_digest = state_digest(torch.get_rng_state())
    identity = {
        "schema": "r02_lai_fit_budget_checkpoint_identity_v1",
        "scientific_use": EXCLUSION, "updates": 1, "role": "FIT",
        "condition": DEFAULT_CONDITION, "seed": seed, "resolved": resolved,
        "source_identity": source_identity or {"synthetic_code_test": True},
    }
    checkpoint_root = output / "checkpoint"
    started = time.perf_counter(); store = CheckpointStore(checkpoint_root, identity)
    try:
        manifest = store.save(model, optimizer, 1, [float(loss.detach())])
    finally:
        store.close()
    phases["checkpoint_save_seconds"] = time.perf_counter() - started
    expected_digest = tensor_state_digest(model)
    restored_model = trainer.TaskModel(resolved, contract)
    restored_optimizer = trainer.build_optimizer(
        restored_model.parameters(), restored_model.encoder.config.learning_rate, "fp32")
    started = time.perf_counter(); restored = CheckpointStore(checkpoint_root, identity)
    try:
        next_step, history = restored.restore(restored_model, restored_optimizer)
    finally:
        restored.close()
    phases["checkpoint_restore_seconds"] = time.perf_counter() - started
    restored_digest = tensor_state_digest(restored_model)
    restore_checks = {
        "model_exact": restored_digest == expected_digest,
        "optimizer_exact": state_digest(restored_optimizer.state_dict()) ==
            expected_optimizer_digest,
        "python_rng_exact": state_digest(random.getstate()) == expected_python_rng_digest,
        "numpy_rng_exact": state_digest(np.random.get_state()) == expected_numpy_rng_digest,
        "torch_rng_exact": state_digest(torch.get_rng_state()) == expected_torch_rng_digest,
    }
    require(next_step == 1 and len(history) == 1 and all(restore_checks.values()),
            "technical checkpoint did not restore the exact post-Adam state")
    checkpoint_manifest = checkpoint_root / (manifest["generation"] + ".json")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "ONE_TECHNICAL_FIT_UPDATE_COMPLETE_AND_RECOVERABLE",
        "scientific_use": EXCLUSION, "selection_eligible": False,
        "role": "FIT", "condition": DEFAULT_CONDITION,
        "select_accessed": False, "evaluate_opened": False,
        "family": resolved["family"], "resolved_configuration": resolved,
        "updates": 1, "threads": threads, "memory_limit_bytes": memory_bytes,
        "people": int(data["person_id"].shape[0]),
        "common_sites": int(data["common_valid"].shape[1]),
        "event_sites": int(data["event_valid"].shape[1]),
        "query_sites": int(data["query_valid"].shape[1]),
        "valid_core_queries_total": int(valid_core_queries.sum()),
        "valid_core_queries_per_mosaic_minimum": int(
            valid_core_queries.sum(axis=1).min()),
        "core_objectives_total": int(core_objectives.sum()),
        "core_objectives_per_mosaic_minimum": int(core_objectives.sum(axis=1).min()),
        "materialized_array_bytes": int(sum(value.nbytes for value in data.values()
                                                if isinstance(value, np.ndarray))),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss_finite": True, "loss_value_withheld_from_budget_decision": True,
        "phase_seconds": phases, "update_seconds": float(sum(phases[name] for name in
            ("forward_seconds", "loss_seconds", "backward_seconds", "adam_seconds"))),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "checkpoint": descriptor(checkpoint_manifest),
        "checkpoint_payload": descriptor(checkpoint_root / manifest["payload"]),
        "checkpoint_model_sha256": expected_digest,
        "checkpoint_restore_exact": True,
        "checkpoint_restore": restore_checks,
        "checkpoint_recovery_boundary": CHECKPOINT_RECOVERY,
        "source_identity": source_identity or {"synthetic_code_test": True},
        "scientific_conclusion": "NOT_ESTABLISHED",
        "limitations": [
            "One regional update does not establish convergence, stopping or total campaign cost.",
            "The checkpoint restores the completed post-Adam state for technical continuation and is excluded from selection.",
            "An interruption before the post-Adam checkpoint is complete repeats this single update; there is no within-update recovery.",
            "This chr22 stress unit does not establish uniform cost across regions or autosomes.",
            "No SELECT or EVALUATE data were opened.",
        ],
    }
    write_fresh(output / "measurement.receipt.json", receipt)
    return receipt


def authenticate_link_source(selected):
    request_spec = descriptor(selected["link_request"]["path"],
                              selected["link_request"]["sha256"])
    receipt_spec = descriptor(selected["link_receipt"]["path"],
                              selected["link_receipt"]["sha256"])
    request = read_json(request_spec)
    receipt = read_json(receipt_spec)
    require(request.get("schema") == "r02_lai_regional_truth_join_request_v2"
            and request.get("operation") == "PREPARE"
            and receipt.get("schema") == "r02_lai_regional_truth_factorized_link_receipt_v1"
            and receipt.get("status") ==
                "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
            and receipt.get("request_sha256") == request_spec["sha256"]
            and request.get("role") == receipt.get("role") == "FIT"
            and request.get("condition") == receipt.get("condition") == DEFAULT_CONDITION
            and request.get("training_authorized") is False
            and request.get("selection_authorized") is False
            and request.get("evaluate_opened") is False
            and receipt.get("role") == "FIT"
            and receipt.get("truth_joined") is True
            and receipt.get("factorized_link") is True
            and receipt.get("training_authorized") is False
            and receipt.get("selection_authorized") is False
            and receipt.get("evaluate_opened") is False,
            "complete factorized source does not match the reconstructed FIT/NONE request")
    return request_spec, receipt_spec


def measure(inventory_path, inventory_sha256, pre_path, pre_sha256, family, output):
    inventory_spec = descriptor(inventory_path, inventory_sha256)
    inventory = read_json(inventory_spec)
    require(inventory.get("schema") == INVENTORY_SCHEMA
            and inventory.get("truth_opened_for_region_choice") is False
            and inventory.get("selection_opened") is False
            and inventory.get("evaluate_opened") is False
            and inventory.get("scientific_use") == EXCLUSION,
            "measurement inventory differs")
    pre_spec = descriptor(pre_path, pre_sha256)
    validate_pre(read_json(pre_spec), inventory_spec, family)
    process_limits = enforced_process_limits()
    output = Path(output)
    require(not output.exists() and output.parent.is_dir(), "fresh family output required")
    output.mkdir(mode=0o700)
    selected = inventory["selections"][family]["selected"]
    request_spec, receipt_spec = authenticate_link_source(selected)
    from r02_lai_neural_input import load_factorized_link
    started = time.perf_counter()
    with Deadline(DEFAULT_TIMEOUT_SECONDS):
        data = load_factorized_link(request_spec["path"], request_spec["sha256"],
                                    receipt_spec["path"], receipt_spec["sha256"],
                                    max_uncompressed_bytes=DEFAULT_MEMORY_BYTES,
                                    action="PREPARE")
        load_seconds = time.perf_counter() - started
        predictor_request = read_json(selected["predictor_request"])
        resolved = resolved_configuration(family, predictor_request)
        source_identity = {
            "inventory": inventory_spec, "pre": pre_spec,
            "factorized_request": request_spec, "factorized_receipt": receipt_spec,
            "predictor_id": selected["predictor_id"],
            "region_id": selected["region_id"], "cost_proxy": selected["cost_proxy"],
            "selection_rule": inventory["selections"][family]["selection_rule"],
        }
        receipt = run_update(data, resolved, output / "update", source_identity=source_identity)
    receipt["factorized_load_seconds"] = load_seconds
    receipt["total_measurement_seconds"] = time.perf_counter() - started
    receipt["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
    receipt["process_limits"] = process_limits
    # Replace the receipt after adding load/total phases; checkpoint remains unchanged.
    final_path = output / "update" / "measurement.receipt.json"
    final_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    for name in ("plan", "plan_sha256", "prepared_root", "task_id", "output"):
        inventory.add_argument("--" + name.replace("_", "-"), required=True)
    run = sub.add_parser("measure")
    for name in ("inventory", "inventory_sha256", "pre", "pre_sha256", "output"):
        run.add_argument("--" + name.replace("_", "-"), required=True)
    run.add_argument("--family", choices=FAMILIES, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.command == "inventory":
        result = build_inventory(args.plan, args.plan_sha256, args.prepared_root,
                                 args.task_id, args.output)
    else:
        result = measure(args.inventory, args.inventory_sha256, args.pre,
                         args.pre_sha256, args.family, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
