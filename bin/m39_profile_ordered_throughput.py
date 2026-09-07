#!/usr/bin/env python3
"""Measure audited loading and optimization throughput on complete TRAIN windows.

The grouped policy estimates a full pass of the same finite input inventory.
Random batches replay the identical sampled pairs, not a population-wide shuffle.
Synthetic targets exercise optimization only; no weights or predictions are saved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import resource
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from m33_safe_bridge_core import require, write_exclusive_json
from m34_prepare_panel_factors import sha256_file
from m39_ordered_batches import estimate_batch_bytes, pack_batch
from m39_ordered_models import OrderedLAIModel, OrderedModelConfig
from m39_profile_ordered_training import (SOURCE_FILES as HISTORICAL_SOURCE_FILES,
                                          authenticate, gradients, recheck_inputs, tensor_digest)
from m39_throughput_sampling import build_sampling_manifest, estimate_grouped_epoch, steps_for_case
from m39_profile_device import (DeviceRuntime, check_warmup_parity, LOGIT_ATOL, LOGIT_RTOL,
                                GRADIENT_ATOL, GRADIENT_RTOL)


SCHEMA = "m39-ordered-throughput-profile-v1"
GPU_SCHEMA = "m39-ordered-throughput-gpu-profile-v1"
SCOPE = "technical_throughput_synthetic_labels_inner_TRAIN_only"
DECISION = "PASS_ORDERED_THROUGHPUT_TECHNICAL_ONLY"
SOURCE_FILES = ("m39_profile_ordered_throughput.py", "m39_throughput_sampling.py",
                "m39_profile_device.py", *HISTORICAL_SOURCE_FILES)
PROFILE_FIELDS = {"schema_version", "scope", "parent_receipt_sha256", "store_case",
                  "store_manifest_sha256", "folds_sha256", "fold", "seed", "device",
                  "warmup_steps", "sample_strata", "anchors_per_stratum", "queries_per_anchor",
                  "learning_rate", "weight_decay", "torch_threads", "max_input_bytes",
                  "max_rss_kib", "max_seconds", "core_sites", "recipe", "cases"}
SMALL_RECIPE = {"width": 32, "depth": 2, "kernels": [3, 7], "dilations": [1, 2],
                "heads": 4, "attention_radius_tokens": 4}
GPU_FIELDS = {"cpu_profile_reference_sha256", "sampling_manifest_sha256", "model_sha256",
              "max_device_bytes", "torch_version", "cuda_version", "gpu_count", "max_workers",
              "logit_atol", "logit_rtol", "gradient_atol", "gradient_rtol"}
PHASE_FIELDS = ("loading_seconds", "zero_grad_seconds", "forward_loss_seconds",
                "backward_seconds", "optimizer_seconds", "explicit_validation_seconds")


def load_profile(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    require(isinstance(cfg, dict), "throughput profile object required")
    gpu = cfg.get("schema_version") == GPU_SCHEMA
    require(set(cfg) == PROFILE_FIELDS | (GPU_FIELDS if gpu else set()), "throughput profile inventory differs")
    require(cfg["schema_version"] in (SCHEMA, GPU_SCHEMA) and cfg["scope"] == SCOPE, "throughput scope differs")
    for name in ("parent_receipt_sha256", "store_manifest_sha256", "folds_sha256"):
        require(isinstance(cfg[name], str) and re.fullmatch("[a-f0-9]{64}", cfg[name]), f"invalid hash: {name}")
    require(cfg["store_case"] == "people_48/radius_1cm" and type(cfg["fold"]) is int
            and cfg["fold"] == 0, "only frozen TRAIN48 radius1 fold0 is in scope")
    for name, low, high in (("seed", 0, 2**32 - 1), ("torch_threads", 1, 2),
                            ("max_input_bytes", 1, 64 * 1024**2),
                            ("max_rss_kib", 1, 6710886), ("max_seconds", 1, 900)):
        require(type(cfg[name]) is int and low <= cfg[name] <= high, f"outside technical envelope: {name}")
    for name, expected in (("warmup_steps", 1), ("sample_strata", 4), ("anchors_per_stratum", 4),
                           ("queries_per_anchor", 2), ("core_sites", 256)):
        require(type(cfg[name]) is int and cfg[name] == expected, f"sampling/core protocol differs: {name}")
    require(cfg["device"] == ("cuda:0" if gpu else "cpu"), "profile/device protocol differs")
    if gpu:
        for name in ("cpu_profile_reference_sha256", "sampling_manifest_sha256", "model_sha256"):
            require(isinstance(cfg[name], str) and re.fullmatch("[a-f0-9]{64}", cfg[name]), "invalid GPU binding hash")
        require(cfg["cpu_profile_reference_sha256"] ==
                "54809d91e5057ef8dde9f3f51f52603830882a76b76a752d8b19e3540f46505c", "CPU reference profile differs")
        require(cfg["sampling_manifest_sha256"] ==
                "af2c580713fb6f1b3d972981559c438469933f80e02562d41901501ed0764be0", "frozen pair manifest differs")
        require(cfg["model_sha256"] ==
                "99670d31f7a4a6f0f1aa3b51f222d77e7c35ba45b0e276410f0b407d56ee98f8", "frozen model differs")
        require(cfg["torch_version"] == "2.12.1+cu126" and cfg["cuda_version"] == "12.6", "CUDA runtime pin differs")
        require(type(cfg["max_device_bytes"]) is int and 0 < cfg["max_device_bytes"] <= 8 * 1024**3,
                "GPU memory ceiling outside envelope")
        require(type(cfg["gpu_count"]) is int and cfg["gpu_count"] == 1
                and type(cfg["max_workers"]) is int and cfg["max_workers"] == 1, "one worker/GPU required")
        require((cfg["logit_atol"], cfg["logit_rtol"], cfg["gradient_atol"], cfg["gradient_rtol"]) ==
                (LOGIT_ATOL, LOGIT_RTOL, GRADIENT_ATOL, GRADIENT_RTOL), "parity tolerances differ")
    require(cfg["learning_rate"] == 0.001 and cfg["weight_decay"] == 0, "optimizer recipe differs")
    require(cfg["recipe"] == SMALL_RECIPE, "only the frozen small recipe is in scope")
    require(all(type(cfg["recipe"][key]) is int for key in ("width", "depth", "heads", "attention_radius_tokens"))
            and all(type(value) is int for key in ("kernels", "dilations") for value in cfg["recipe"][key]),
            "recipe counts must be integers")
    require(isinstance(cfg["cases"], list) and len(cfg["cases"]) == 6, "exactly six throughput cases required")
    seen = set()
    for case in cfg["cases"]:
        require(isinstance(case, dict) and set(case) == {"id", "family", "policy", "batch_size", "arm"},
                "throughput case fields differ")
        require(case["family"] in ("cnn", "attention") and case["arm"] == "real", "case family/arm differs")
        require(type(case["batch_size"]) is int and (case["policy"], case["batch_size"]) in
                (("grouped", 1), ("grouped", 2), ("random", 2)), "batch policy differs")
        expected = "{family}-small-{policy}-b{batch_size}-{arm}".format(**case)
        require(case["id"] == expected and expected not in seen, "case identity differs")
        seen.add(expected)
    return cfg


def check_limits(started: float, cfg: dict) -> int:
    require(time.monotonic() - started <= cfg["max_seconds"], "case wall time exceeded")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(rss <= cfg["max_rss_kib"], "case RSS exceeded")
    return int(rss)


def measure_step(store, model, optimizer, row: dict, case: dict, cfg: dict,
                 *, step_index: int, warmup: bool, case_started: float) -> dict:
    runtime = DeviceRuntime(cfg["device"], cfg.get("max_device_bytes", 0))
    started = runtime.tick()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    tick = runtime.tick()
    pairs = row["pairs"]
    plan = estimate_batch_bytes(store, pairs)
    batch = pack_batch(store, pairs, max_input_bytes=cfg["max_input_bytes"], device="cpu")
    loading_seconds = runtime.elapsed(tick)
    transfer_seconds = 0.0
    if runtime.is_cuda:
        tick = runtime.tick()
        batch = runtime.transfer(batch)
        transfer_seconds = runtime.elapsed(tick)
    tick = runtime.tick()
    require(len(pairs) == case["batch_size"], "sampled batch size differs")
    lengths = [store.window_bounds(j)[1] - store.window_bounds(j)[0] for _, j in pairs]
    require(lengths == row["anchor_lengths"], "sampling lengths differ from source")
    require(batch["site_mask"].sum(dim=1).tolist() == lengths, "full windows were not retained")
    digest_before = tensor_digest(batch)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    check_limits(case_started, cfg)
    runtime.memory()
    validation_seconds = runtime.elapsed(tick)

    tick = runtime.tick()
    optimizer.zero_grad(set_to_none=True)
    zero_grad_seconds = runtime.elapsed(tick)
    # The ordinal schedule never depends on a genotype, ID, or biological label.
    labels = (torch.arange(case["batch_size"], device=cfg["device"]) + step_index) % 6
    tick = runtime.tick()
    logits = model(batch, arm=case["arm"], chunked=True)
    loss = F.cross_entropy(logits, labels)
    forward_seconds = runtime.elapsed(tick)
    tick = runtime.tick()
    require(bool(torch.isfinite(loss)), "nonfinite synthetic loss")
    check_limits(case_started, cfg)
    runtime.memory()
    validation_seconds += runtime.elapsed(tick)

    tick = runtime.tick()
    loss.backward()
    backward_seconds = runtime.elapsed(tick)
    tick = runtime.tick()
    norms = gradients(model)
    check_limits(case_started, cfg)
    runtime.memory()
    validation_seconds += runtime.elapsed(tick)
    tick = runtime.tick()
    optimizer.step()
    optimizer_seconds = runtime.elapsed(tick)
    tick = runtime.tick()
    updated = {}
    for name, parameter in model.named_parameters():
        require(bool(torch.isfinite(parameter).all()), f"nonfinite parameter: {name}")
        group = name.split(".")[0]
        updated[group] = updated.get(group, 0) + int(not torch.equal(before[name], parameter))
    require(updated and all(value > 0 for value in updated.values()), "component weights unchanged")
    optimizer_bytes = sum(t.numel() * t.element_size() for state in optimizer.state.values()
                          for t in state.values() if isinstance(t, torch.Tensor))
    require(optimizer_bytes > 0, "optimizer state was not allocated")
    require(tensor_digest(batch) == digest_before, "input batch was mutated")
    rss = check_limits(case_started, cfg)
    tensor_bytes = sum(t.numel() * t.element_size() for t in batch.values())
    padded_sites = int(batch["channels"].shape[-2])
    device_memory = runtime.memory()
    validation_seconds += runtime.elapsed(tick)
    del loss, logits, batch, before
    elapsed = runtime.elapsed(started)
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    record = {"step": step_index, "warmup": warmup, "stratum": row["stratum"],
              "pairs": pairs, "full_window_sites": lengths, "padded_sites_per_row": padded_sites,
              "true_site_tokens": sum(lengths), "padded_site_tokens": len(pairs) * padded_sites,
              "allocation_plan": plan, "tensor_bytes": tensor_bytes, "tensor_sha256": digest_before,
              "loading_seconds": loading_seconds, "zero_grad_seconds": zero_grad_seconds,
              "forward_loss_seconds": forward_seconds, "backward_seconds": backward_seconds,
              "optimizer_seconds": optimizer_seconds, "explicit_validation_seconds": validation_seconds,
              "end_to_end_seconds": elapsed, "optimizer_state_bytes": optimizer_bytes,
              "gradient_norm_by_component": norms, "updated_tensors_by_component": updated,
              "peak_rss_kib": rss,
              "process_cpu_seconds": ((usage_after.ru_utime + usage_after.ru_stime) -
                                      (usage_before.ru_utime + usage_before.ru_stime))}
    if runtime.is_cuda:
        record.update(transfer_seconds=transfer_seconds, device_memory=device_memory)
    record["orchestration_seconds"] = max(0.0, elapsed - sum(record[name] for name in PHASE_FIELDS)
                                           - transfer_seconds)
    return record


def summarize_steps(rows: list[dict], *, policy: str, batch_size: int, sampling: dict) -> dict:
    measured = [row for row in rows if not row["warmup"]]
    require(measured, "no measured steps completed")
    elapsed = sum(row["end_to_end_seconds"] for row in measured)
    require(np.isfinite(elapsed) and elapsed > 0, "invalid measured duration")
    examples = sum(len(row["pairs"]) for row in measured)
    result = {"measured_steps": len(measured), "measured_query_anchor_pairs": examples,
              "measured_end_to_end_seconds": elapsed, "query_anchor_pairs_per_second": examples / elapsed,
              "phase_seconds": {key: sum(row[key] for row in measured) for key in PHASE_FIELDS},
              "step_seconds_min": min(row["end_to_end_seconds"] for row in measured),
              "step_seconds_max": max(row["end_to_end_seconds"] for row in measured),
              "true_site_tokens": sum(row["true_site_tokens"] for row in measured),
              "padded_site_tokens": sum(row["padded_site_tokens"] for row in measured),
              "grouped_fullpass_estimate": None}
    if any("transfer_seconds" in row for row in measured):
        require(all("transfer_seconds" in row for row in measured), "device timing inventory differs")
        result["phase_seconds"]["transfer_seconds"] = sum(row["transfer_seconds"] for row in measured)
    if policy == "grouped":
        estimate = estimate_grouped_epoch(
            [{**row, "e2e_seconds": row["end_to_end_seconds"]} for row in measured], sampling, batch_size)
        result["grouped_fullpass_estimate"] = {
            **estimate, "seconds": estimate["estimated_grouped_pass_seconds"],
            "population_query_anchor_pairs": estimate["full_pair_count"],
            "scope": ("same_CUDA_small_recipe_grouped_policy_audited_steps_only" if
                      "transfer_seconds" in measured[0] else "same_CPU_small_recipe_grouped_policy_audited_steps_only"),
            "confidence_interval": None, "convergence_time_estimated": False}
    else:
        require(policy == "random", "unknown batching policy")
        result["random_scope"] = "conditional_replay_of_same_sampled_pairs_not_population_shuffle_estimate"
    return result


def run_case(store_dir: Path, parent_path: Path, folds_path: Path, profile_path: Path,
             case_id: str, output_dir: Path, source_commit: str) -> dict:
    started = time.monotonic()
    cfg = load_profile(profile_path)
    require(re.fullmatch("[a-f0-9]{40}", source_commit), "full source commit required")
    require(not output_dir.exists(), "output already exists")
    matches = [case for case in cfg["cases"] if case["id"] == case_id]
    require(len(matches) == 1, "unknown throughput case")
    case = matches[0]
    torch.set_num_threads(cfg["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    runtime = DeviceRuntime(cfg["device"], cfg.get("max_device_bytes", 0))
    device_info = runtime.configure()
    paths = {profile_path: sha256_file(profile_path), parent_path: cfg["parent_receipt_sha256"],
             folds_path: cfg["folds_sha256"], store_dir / "manifest.json": cfg["store_manifest_sha256"]}
    code_hashes = {}
    for name in SOURCE_FILES:
        source = Path(__file__).with_name(name)
        code_hashes[name] = sha256_file(source)
        paths[source] = code_hashes[name]
    store, source_manifest = authenticate(store_dir, parent_path, folds_path, cfg)
    if runtime.is_cuda:
        require(code_hashes["m39_ordered_models.py"] == cfg["model_sha256"], "model source changed")
    sampling = build_sampling_manifest(store, cfg["seed"])
    measured_rows = steps_for_case(sampling, case["batch_size"], case["policy"])
    require(len(measured_rows) == 32 // case["batch_size"], "sampled batch inventory differs")
    warmup = sampling["warmup"]
    warmup_row = {"stratum": None, "pairs": warmup["pairs"][:case["batch_size"]],
                  "anchor_lengths": warmup["anchor_lengths"][:case["batch_size"]]}
    recipe = dict(cfg["recipe"])
    recipe["kernels"], recipe["dilations"] = tuple(recipe["kernels"]), tuple(recipe["dilations"])
    model_config = OrderedModelConfig(family=case["family"], **recipe, dropout=0,
                                     core_sites=cfg["core_sites"], checkpoint_chunks=True)
    model = OrderedLAIModel(model_config).to(cfg["device"]).train()
    optimizer_options = {"foreach": False, "fused": False} if runtime.is_cuda else {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"], **optimizer_options)
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    steps_dir = output_dir / "steps"
    steps_dir.mkdir(mode=0o700)
    write_exclusive_json(output_dir / "sampling-manifest.json", sampling)
    sampling_sha = sha256_file(output_dir / "sampling-manifest.json")
    parity = None
    if runtime.is_cuda:
        require(sampling_sha == cfg["sampling_manifest_sha256"], "frozen sampled pairs changed")
        tick = time.monotonic()
        parity_batch = pack_batch(store, warmup_row["pairs"], max_input_bytes=cfg["max_input_bytes"], device="cpu")
        parity_digest = tensor_digest(parity_batch)
        parity = check_warmup_parity(model, parity_batch, torch.arange(case["batch_size"]) % 6, runtime.device)
        require(tensor_digest(parity_batch) == parity_digest, "parity input batch mutated")
        parity.update(seconds=time.monotonic() - tick, pairs=warmup_row["pairs"],
                      full_window_sites=warmup_row["anchor_lengths"], tensor_sha256=parity_digest,
                      device_memory=runtime.memory(), optimizer_update_parity=False)
        write_exclusive_json(output_dir / "device-parity.json", parity)
        del parity_batch
        runtime.reset_measurement_peaks()
    prepared_seconds = time.monotonic() - started
    rows = []
    step_receipt_seconds = 0.0
    for index, row in enumerate([warmup_row, *measured_rows]):
        record = measure_step(store, model, optimizer, row, case, cfg,
                              step_index=index, warmup=index == 0, case_started=started)
        rows.append(record)
        tick = time.monotonic()
        write_exclusive_json(steps_dir / f"step-{index:04d}.json", record)
        step_receipt_seconds += time.monotonic() - tick
        print(json.dumps({"event": "step_complete", "case": case_id, "step": index,
                          "warmup": record["warmup"], "seconds": record["end_to_end_seconds"],
                          "peak_rss_kib": record["peak_rss_kib"]}), flush=True)
    summary = summarize_steps(rows, policy=case["policy"], batch_size=case["batch_size"],
                              sampling=sampling)
    tick = time.monotonic()
    recheck_inputs(store_dir, source_manifest, paths)
    require(sha256_file(output_dir / "sampling-manifest.json") == sampling_sha, "sampling manifest changed")
    closing_seconds = time.monotonic() - tick
    rss = check_limits(started, cfg)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {"schema_version": cfg["schema_version"], "decision": DECISION, "case": case,
              "architecture": asdict(model_config), "parameter_count": sum(p.numel() for p in model.parameters()),
              "receptive_radius_tokens": model_config.receptive_radius_tokens,
              "store_shape": list(store.shape), "sampling_manifest_sha256": sampling_sha,
              "steps": rows, "summary": summary, "prepared_seconds": prepared_seconds,
              "step_receipt_write_seconds": step_receipt_seconds,
              "closing_audit_seconds": closing_seconds, "elapsed_seconds": time.monotonic() - started,
              "resources": {"peak_rss_kib": rss, "user_cpu_seconds": usage.ru_utime,
                            "system_cpu_seconds": usage.ru_stime},
              "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                          "device": cfg["device"], "threads": torch.get_num_threads(), "dtype": "float32"},
              "provenance": {"source_commit": source_commit, "source_sha256": code_hashes,
                             "profile_sha256": paths[profile_path], "parent_sha256": cfg["parent_receipt_sha256"],
                             "manifest_sha256": cfg["store_manifest_sha256"], "folds_sha256": cfg["folds_sha256"]},
              "scope": {"optimization_executed": True, "biological_training": False,
                        "label_source": "(step + batch_row) mod 6; artificial",
                        "truth_opened": False, "predictions_opened": False, "feature_roles": ["TRAIN"],
                        "weights_saved": False, "predictions_saved": False, "accuracy_evaluated": False,
                        "new_cloud_instances": 0, "cold_storage_benchmark": False,
                        "model_internal_validation_in_forward": True,
                        "audits_in_end_to_end_timing": True, "fullpass_excludes_one_time_setup": True,
                        "step_receipt_io_outside_step_timing": True,
                        "partial_step_receipts_are_not_PASS": True,
                        "medium_or_GPU_extrapolation": False}}
    if runtime.is_cuda:
        report["runtime"].update(device_info)
        report["runtime"]["optimizer_options"] = optimizer_options
        report["resources"]["device_memory"] = runtime.memory()
        report["device_parity"] = parity
        report["provenance"]["cpu_profile_reference_sha256"] = cfg["cpu_profile_reference_sha256"]
        report["scope"].update(gpu_measured=True, same_image_cpu_gradient_reference=True,
                              historical_cpu_concurrency_differs=True, transfer_in_end_to_end=True,
                              device_phase_timers_synchronized=True)
    write_exclusive_json(output_dir / "profile.json", report)
    print(json.dumps({"case": case_id, "decision": DECISION,
                      "elapsed_seconds": report["elapsed_seconds"], "peak_rss_kib": rss}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store-dir", "parent-receipt", "folds", "profile-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    cfg = load_profile(args.profile_config)
    def timeout(_signal, _frame):
        raise TimeoutError("throughput case time limit reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cfg["max_seconds"])
    run_case(args.store_dir.resolve(), args.parent_receipt.resolve(), args.folds.resolve(),
             args.profile_config.resolve(), args.case_id, args.output_dir, args.source_commit)


if __name__ == "__main__":
    main()
