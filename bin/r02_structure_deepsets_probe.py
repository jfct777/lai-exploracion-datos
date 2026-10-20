#!/usr/bin/env python3
"""One bounded real-FIT Deep Sets forward/backward, without an optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

import r02_context_train as trainer
from r02_context_autosomal_dataset import open_autosomal


SCHEMA = "r02_structure_deepsets_fit_probe_plan_v1"
CHROM22 = "ALLELE_POSITION_CHROM22_V1"
RARE_MEAN = "MEAN_WITH_EXPLICIT_BURDEN"
PREFIX_MEAN = "PREFIX_RANGE_MEAN_V1"
CONTEXT_EXECUTION = {"query_block_size": 4096, "max_mask_elements": 324584,
                     "checkpoint_blocks": True, "local_mean_mode": PREFIX_MEAN}
CODE_FILES = (
    "r02_structure_deepsets_probe.py", "r02_context_train.py", "genomic_context_runtime.py",
    "r02_context_autosomal_pool.py", "r02_context_checkpoint.py", "r02_adam_fp64_state.py",
    "r02_local_attention_blocks.py", "r02_k03_neural_target.py",
    "r02_structure_identity.py",
    "r02_context_autosomal_dataset.py", "r02_event_mmap_reader.py",
    "r02_genomic_event_inputs.py", "r02_genomic_event_cache.py",
    "r02_genomic_pair_evidence.py", "r02_structure_increment.py",
    "r02_structure_input_plan.py", "r02_structure_materialize.py", "r02_structure_regions.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, digest):
    require(sha256(path) == digest, "Deep Sets plan SHA256 differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA and
            plan.get("status") == "ROOT_APPROVED_ONE_TECHNICAL_PASS_NOT_EXECUTED",
            "unapproved Deep Sets probe plan")
    source, model, limits = plan.get("input", {}), plan.get("model", {}), plan.get("resources", {})
    require(source.get("stage") == "FIT" and source.get("chromosome") == 3 and
            source.get("person_batch_index") == 0 and source.get("expected_people") == 8 and
            source.get("expected_pairs") == 28 and source.get("tile_core_bp") == [86049848, 96359743],
            "Deep Sets FIT input scope differs")
    require(model == {"family": "structure_deep_sets", "common_radius_cm": 1.0,
                      "rare_radius_cm": None, "width": 64, "depth": 2, "heads": None,
                      "dropout": 0.0, "learning_rate": 0.001,
                      "learning_rate_role": "INERT_CONFIGURATION_FIELD_NO_OPTIMIZER",
                      "rare_pooling": RARE_MEAN, "locus_descriptor": CHROM22,
                      "arm": "RARE", "seed": 1701},
            "Deep Sets model differs from approved probe")
    require(plan.get("context_execution") == CONTEXT_EXECUTION,
            "Deep Sets prefix execution differs")
    require(limits.get("cpu") == 1 and limits.get("memory_gib") == 6 and
            limits.get("wall_seconds") == 900 and limits.get("output_max_mib") == 10 and
            limits.get("network") == "none" and limits.get("root_filesystem") == "read_only",
            "Deep Sets resource limits differ")
    require(limits.get("host_memory_check_required") is True and
            limits.get("known_allocation_preflight_required") is True,
            "Deep Sets memory preflights differ")
    for key in ("request_path", "request_sha256", "target_path", "target_sha256",
                "target_receipt_path", "target_receipt_sha256"):
        require(source.get(key), "missing authenticated input " + key)
    return source, model, limits


def subset_target(path, expected_sha256, batch_ids, expected_pairs):
    require(sha256(path) == expected_sha256, "FIT target SHA256 differs")
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == {"person_id", "component_id", "role", "pairs", "target",
                                     "pair_weight", "joint_marker_count"},
                "FIT target arrays differ")
        ids = loaded["person_id"].astype(str)
        require(tuple(ids[:len(batch_ids)]) == tuple(batch_ids), "FIT target/person batch order differs")
        pairs = np.asarray(loaded["pairs"])
        keep = (pairs[:, 0] < len(batch_ids)) & (pairs[:, 1] < len(batch_ids))
        result = {key: np.asarray(loaded[key][keep]) for key in
                  ("pairs", "target", "pair_weight", "joint_marker_count")}
    require(len(result["pairs"]) == expected_pairs and result["pairs"].shape == (expected_pairs, 2) and
            np.isfinite(result["target"]).all() and np.isfinite(result["pair_weight"]).all() and
            (result["pair_weight"] > 0).all(), "FIT within-batch pair target differs")
    return result


def forward_backward_measurement(tile, pair_data, model_plan, context_execution=CONTEXT_EXECUTION):
    resolved = {key: model_plan[key] for key in
                ("family", "common_radius_cm", "rare_radius_cm", "width", "depth", "heads",
                 "dropout", "learning_rate", "rare_pooling", "locus_descriptor")}
    contract = {"objective": "structure", "common_features": int(tile["common_features"].shape[-1]),
                "event_metadata_features": int(tile["event_metadata"].shape[-1]),
                "limits": {"max_common_sites": int(tile["common_features"].shape[1]),
                           "max_event_sites": int(tile["event_metadata"].shape[1]),
                           "max_targets": int(len(pair_data["target"]))},
                "context_execution": context_execution}
    data = dict(tile)
    data.update(pair_data)
    torch.set_num_threads(1)
    torch.manual_seed(model_plan["seed"])
    model_started = time.monotonic()
    model = trainer.TaskModel(resolved, contract)
    model_seconds = time.monotonic() - model_started
    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    forward_started = time.monotonic()
    prediction = model(data, model_plan["arm"])
    loss = trainer.training_loss(prediction, data, "structure")
    forward_seconds = time.monotonic() - forward_started
    require(bool(torch.isfinite(loss)) and bool(torch.isfinite(prediction).all()),
            "nonfinite Deep Sets prediction/loss")
    backward_started = time.monotonic()
    loss.backward()
    backward_seconds = time.monotonic() - backward_started
    require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()), "nonfinite Deep Sets gradient")
    unchanged = all(torch.equal(parameter.detach(), before[name])
                    for name, parameter in model.named_parameters())
    require(unchanged, "parameters changed without optimizer")
    diagnostics = trainer.aggregate_norm_diagnostics(model, prediction, None, before)
    first_event_weight = model.encoder.event_encoder[0].weight
    dose_column = int(tile["event_metadata"].shape[-1])
    require(first_event_weight.grad is not None and dose_column == 12 and
            dose_column < first_event_weight.grad.shape[1],
            "event dose input column differs from the frozen 12-metadata layout")
    dose_gradient_l2 = float(torch.linalg.vector_norm(
        first_event_weight.grad[:, dose_column].detach().double()))
    require(math.isfinite(dose_gradient_l2), "event dose-column gradient is nonfinite")
    required_groups = {"common_encoder", "event_encoder", "person_rho", "pair_head"}
    require(required_groups <= set(diagnostics["modules"]) and
            all(math.isfinite(diagnostics["modules"][group]["gradient_l2"]) and
                diagnostics["modules"][group]["gradient_l2"] > 0
                for group in required_groups), "Deep Sets required module gradient is absent/nonpositive")
    require(all(values["update_l2"] == 0 for values in diagnostics["modules"].values()),
            "parameter update occurred")
    result = {"loss": float(loss.detach()), "model_parameters": sum(p.numel() for p in model.parameters()),
              "model_seconds": model_seconds, "forward_seconds": forward_seconds,
              "backward_seconds": backward_seconds, "parameters_changed": False,
              "optimizer_created": False, "optimizer_step": False, "diagnostics": diagnostics}
    result["event_encoder_first_weight"] = {
        "input_layout": "12_EVENT_METADATA_THEN_DOSE_THEN_3_LOG1P_BURDENS_THEN_COMMON_CONTEXT_COUNT",
        "dose_column_zero_based": dose_column,
        "dose_gradient_l2": dose_gradient_l2,
        "interpretation": "TECHNICAL_GRADIENT_CONNECTIVITY_ONLY"}
    return result


def known_allocation(tile, model_plan):
    """Conservative sum of explicit arrays and same-shape gradient storage.

    This is an admission calculation, not a prediction of allocator peak RSS.
    It deliberately counts every named activation plus one same-shape gradient.
    """
    people, common = tile["common_features"].shape[:2]
    events = tile["event_metadata"].shape[1]
    width, depth = model_plan["width"], model_plan["depth"]
    event_input_width = tile["event_metadata"].shape[-1] + 4 + width + 1
    arrays = {
        "loaded_numpy_arrays": sum(value.nbytes for value in tile.values()
                                   if isinstance(value, np.ndarray)),
        "common_encoder_output_fp32": people * common * width * 4,
        "common_encoder_activation_fp32": people * common * width * 4,
        "prefix_values_fp64": people * (common + 1) * width * 8,
        "prefix_counts_int64": people * (common + 1) * 8,
        "event_common_mean_fp32": people * events * width * 4,
        "event_common_count_int64": people * events * 8,
        "event_encoder_input_fp32": people * events * event_input_width * 4,
        "event_encoder_hidden_and_gelu_fp32": people * events * width * depth * 2 * 4,
    }
    differentiable = sum(value for key, value in arrays.items()
                         if key not in ("loaded_numpy_arrays", "prefix_counts_int64",
                                        "event_common_count_int64"))
    known = sum(arrays.values()) + differentiable
    return {"people": people, "common_site_slots": common, "event_site_slots": events,
            "width": width, "depth": depth, "named_bytes": arrays,
            "same_shape_gradient_bytes": differentiable,
            "known_total_bytes": sum(arrays.values()) + differentiable,
            "interpretation": "ADMISSION_SUM_NOT_PEAK_RSS_PREDICTION"}


def cgroup_memory_limit_bytes():
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.exists():
            value = path.read_text().strip()
            if value != "max":
                return int(value)
    return None


def run(plan_path, plan_sha256, output):
    started = time.monotonic()
    plan = read_json(plan_path, plan_sha256)
    source, model_plan, limits = validate_plan(plan)
    require(str(Path(output)) == limits.get("output_path"),
            "Deep Sets output path differs from approved plan")
    output_path = Path(output)
    trainer.configure_torch_cache(output_path.parent / f".{output_path.name}.torch-cache")
    # Fail on PyTorch startup before authenticating or reading the human tile.
    torch.use_deterministic_algorithms(True)
    source_hashes = {name: sha256(source[name + "_path"])
                     for name in ("request", "target", "target_receipt")}
    require(all(source_hashes[name] == source[name + "_sha256"] for name in source_hashes),
            "Deep Sets input SHA256 differs")
    receipt = json.loads(Path(source["target_receipt_path"]).read_text())
    require(receipt.get("role") == "FIT" and receipt.get("evaluate_targets_computed") is False and
            receipt.get("training_allowed") is False, "FIT target receipt role/state differs")
    dataset = open_autosomal(source["request_path"], source["request_sha256"])
    request = dataset.request
    require(request["stage"] == "FIT" and request["common_radius_cm"] == 1.0 and
            request["rare_radius_cm"] is None and request["attention_depth"] == 0,
            "request is not the existing Deep Sets radius-1.0 geometry")
    chromosome = str(source["chromosome"])
    require(source["tile_core_bp"] in request["tiles"][chromosome], "Deep Sets tile is absent")
    ids, roles, components = dataset.person_batches[source["person_batch_index"]]
    require(len(ids) == source["expected_people"] and set(roles) == {"FIT"}, "FIT batch differs")
    read_started = time.monotonic()
    reader = dataset._reader(chromosome)
    start_bp, end_bp = source["tile_core_bp"]
    view = reader.load_window(list(ids), start_bp, end_bp, request["block_id"], request["exclusion_scope"],
                              request["common_radius_cm"], request["rare_radius_cm"], request["attention_depth"])
    read_seconds = time.monotonic() - read_started
    require(tuple(map(str, view["person_id"])) == ids and tuple(map(str, view["role"])) == roles and
            tuple(map(str, view["component_id"])) == components, "reader person identity/order differs")
    tile = trainer.autosomal_tile(view)
    pairs = subset_target(source["target_path"], source["target_sha256"], ids, source["expected_pairs"])
    allocation = known_allocation(tile, model_plan)
    memory_limit = limits["memory_gib"] * 2**30
    observed_cgroup_limit = cgroup_memory_limit_bytes()
    cgroup_matches = (observed_cgroup_limit is not None and observed_cgroup_limit <= memory_limit and
                      observed_cgroup_limit >= memory_limit - 4096)
    allocation_fits = allocation["known_total_bytes"] <= memory_limit
    if not (cgroup_matches and allocation_fits):
        final_hashes = {name: sha256(source[name + "_path"])
                        for name in ("request", "target", "target_receipt")}
        require(final_hashes == source_hashes, "Deep Sets input changed during preflight")
        output = Path(output)
        require(not output.exists(), "Deep Sets output already exists")
        output.mkdir(parents=True)
        result = {
            "schema": "r02_structure_deepsets_fit_probe_receipt_v1",
            "status": "FIT_DEEPSETS_CANCELLED_BEFORE_MODEL_MEMORY_PREFLIGHT",
            "plan": {"path": str(Path(plan_path).resolve()), "sha256": plan_sha256},
            "source_sha256": source_hashes,
            "scope": {"stage": "FIT", "chromosome": int(chromosome),
                      "tile_core_bp": [start_bp, end_bp], "person_batch_index": 0,
                      "people": len(ids), "pairs": len(pairs["pairs"]),
                      "persons_emitted": False, "select_opened": False,
                      "evaluate_opened": False},
            "reader": {"common_site_slots": int(tile["common_valid"].shape[1]),
                       "event_site_slots": int(tile["event_valid"].shape[1]),
                       "common_valid_total": int(tile["common_valid"].sum()),
                       "event_valid_total": int(tile["event_valid"].sum()),
                       "read_seconds": read_seconds},
            "known_allocation_preflight": allocation,
            "context_execution": plan["context_execution"],
            "resources": {"elapsed_seconds": time.monotonic() - started,
                          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          "cpu_limit": 1, "memory_limit_gib": 6,
                          "cgroup_memory_limit_bytes": observed_cgroup_limit,
                          "wall_limit_seconds": 900},
            "execution": {"torch_imported": True, "model_created": False,
                          "technical_backward_executed": False, "optimizer_created": False,
                          "optimizer_step": False, "weights_modified": False,
                          "training_started": False, "select_opened": False,
                          "evaluate_opened": False},
            "cancellation": {"cgroup_matches_6gib": cgroup_matches,
                             "known_allocation_fits": allocation_fits},
            "allowed_conclusion": "Reader counts and the explicit known-allocation admission calculation for this exact FIT tile.",
            "not_allowed": ["model memory or gradient connectivity", "complete 22-autosome duration",
                            "training convergence", "SELECT/EVALUATE performance", "biological information"]}
        (output / "receipt.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return result
    measured = forward_backward_measurement(tile, pairs, model_plan, plan["context_execution"])
    final_hashes = {name: sha256(source[name + "_path"])
                    for name in ("request", "target", "target_receipt")}
    require(final_hashes == source_hashes, "Deep Sets input changed during probe")
    output = Path(output)
    require(not output.exists(), "Deep Sets output already exists")
    output.mkdir(parents=True)
    code_dir = Path(__file__).resolve().parent
    result = {"schema": "r02_structure_deepsets_fit_probe_receipt_v1",
              "status": "FIT_DEEPSETS_FORWARD_BACKWARD_COMPLETE_NO_OPTIMIZER",
              "plan": {"path": str(Path(plan_path).resolve()), "sha256": plan_sha256},
              "source_sha256": source_hashes,
              "scope": {"stage": "FIT", "chromosome": int(chromosome),
                        "tile_core_bp": [start_bp, end_bp], "person_batch_index": 0,
                        "people": len(ids), "pairs": len(pairs["pairs"]), "persons_emitted": False,
                        "select_opened": False, "evaluate_opened": False},
              "reader": {"common_site_slots": int(tile["common_valid"].shape[1]),
                         "event_site_slots": int(tile["event_valid"].shape[1]),
                         "common_valid_total": int(tile["common_valid"].sum()),
                         "event_valid_total": int(tile["event_valid"].sum()),
                         "common_core_total": int(tile["common_core"].sum()),
                         "event_core_total": int(tile["event_core"].sum()),
                         "read_seconds": read_seconds},
              "known_allocation_preflight": allocation,
              "context_execution": plan["context_execution"],
              "model": {key: value for key, value in model_plan.items()
                        if key != "learning_rate_role"},
              "measurement": measured,
              "resources": {"elapsed_seconds": time.monotonic() - started,
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            "cpu_limit": 1, "memory_limit_gib": 6,
                            "cgroup_memory_limit_bytes": observed_cgroup_limit,
                            "wall_limit_seconds": 900},
              "execution": {"torch_imported": True, "model_created": True,
                            "technical_backward_executed": True,
                            "optimizer_created": False, "optimizer_step": False,
                            "weights_modified": False, "training_started": False,
                            "select_opened": False, "evaluate_opened": False},
              "code_sha256": {name: sha256(code_dir / name) for name in CODE_FILES},
              "allowed_conclusion": "Memory, time and finite gradient connectivity for this exact dense FIT Deep Sets window.",
              "not_allowed": ["complete 22-autosome update duration", "training convergence",
                              "SELECT/EVALUATE performance", "biological information"]}
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    require(len(encoded.encode()) <= limits["output_max_mib"] * 2**20, "Deep Sets receipt exceeds output limit")
    (output / "receipt.json").write_text(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.plan, args.plan_sha256, args.output)
    summary = {"status": result["status"],
               "peak_rss_bytes": result["resources"]["peak_rss_bytes"]}
    if "measurement" in result:
        summary["loss"] = result["measurement"]["loss"]
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
