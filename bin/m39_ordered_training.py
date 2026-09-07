#!/usr/bin/env python3
"""Budgeted ordered-anchor training; selection reads development only.

The same candidate can be trained as COMMON, POOLED, REAL or a reference-link
SHAM. This module does not create a cloud machine, read SCORE, infer rare phase,
or interpolate a chromosome-wide ancestry sequence from sparse anchors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from m33_safe_bridge_core import write_deterministic_npz, write_exclusive_json
from m39_anchor_screen import metrics, sha256
from m39_ordered_batches import pack_batch
from m39_ordered_context import OrderedContextStore
from m39_ordered_models import OrderedLAIModel, OrderedModelConfig, STATE_NAMES
from m39_ordered_training_data import (apply_reference_link_sham, bind_development,
                                      fixed_anchor_subset, paired_batches,
                                      reference_link_permutation, require)
from m39_profile_device import DeviceRuntime

SCHEMA = "m39-ordered-anchor-training-v1"
ARMS = ("common", "pooled", "real", "sham")
SOURCE_FILES = ("m39_ordered_training.py", "m39_ordered_training_data.py", "m39_ordered_models.py",
                "m39_ordered_batches.py", "m39_ordered_context.py", "m39_profile_device.py",
                "m39_anchor_screen.py", "m39_carrier_models.py", "m33_safe_bridge_core.py",
                "m34_prepare_panel_factors.py", "m34_generate_mosaics.py")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    required = {"schema_version", "case_id", "scope", "arm", "model", "seed", "pair_seed",
                "anchor_seed", "sham_seed", "anchor_count", "steps", "evaluate_every_steps",
                "batch_size", "learning_rate", "weight_decay", "gradient_clip_norm",
                "device", "cpu_threads", "max_input_bytes", "max_device_bytes", "max_rss_bytes",
                "max_runtime_seconds", "train_manifest_sha256", "select_manifest_sha256",
                "development_sha256", "selection_metric", "paired_budget_id"}
    require(set(cfg) == required and cfg["schema_version"] == SCHEMA, "training config fields differ")
    require(cfg["scope"] == "exploratory_chr22_R0_development_anchors_only"
            and cfg["arm"] in ARMS and cfg["selection_metric"] == "brier", "training scope/arm differs")
    for key in ("case_id", "paired_budget_id"):
        value = cfg[key]
        require(isinstance(value, str) and value and len(value) <= 160
                and set(value) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"),
                f"invalid {key}")
    for key in ("seed", "pair_seed", "anchor_seed", "sham_seed"):
        require(type(cfg[key]) is int and 0 <= cfg[key] < 2**63, f"invalid {key}")
    for key in ("anchor_count", "steps", "evaluate_every_steps", "batch_size", "cpu_threads",
                "max_input_bytes", "max_rss_bytes", "max_runtime_seconds"):
        require(type(cfg[key]) is int and cfg[key] > 0, f"invalid positive integer {key}")
    require(cfg["evaluate_every_steps"] <= cfg["steps"], "checkpoint interval exceeds budget")
    for key in ("learning_rate", "gradient_clip_norm"):
        require(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] > 0,
                f"invalid {key}")
    require(type(cfg["weight_decay"]) in (int, float) and np.isfinite(cfg["weight_decay"])
            and cfg["weight_decay"] >= 0, "invalid weight decay")
    model = OrderedModelConfig(**cfg["model"])
    require(model.dropout == 0, "audited chunked training currently requires zero dropout")
    require(set(cfg["model"]) == set(asdict(model)), "all architecture values must be explicit")
    DeviceRuntime(cfg["device"], cfg["max_device_bytes"])
    for key in ("train_manifest_sha256", "select_manifest_sha256", "development_sha256"):
        require(isinstance(cfg[key], str) and len(cfg[key]) == 64
                and set(cfg[key]) <= set("0123456789abcdef"), f"invalid {key}")
    return cfg


def _limits(cfg: dict, runtime: DeviceRuntime, started: float) -> None:
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= cfg["max_rss_bytes"],
            "training RSS ceiling exceeded")
    require(time.monotonic() - started <= cfg["max_runtime_seconds"], "training time ceiling exceeded")
    runtime.memory()


def _batch(store, pairs, cfg, runtime, permutation):
    batch = pack_batch(store, pairs, max_input_bytes=cfg["max_input_bytes"])
    if permutation is not None:
        batch = apply_reference_link_sham(batch, store, pairs, permutation)
    return runtime.transfer(batch)


def predict(model, store, anchors, cfg, runtime, permutation, *, started):
    """Evaluate every role person on each declared anchor, never a sparse subset of people."""
    probabilities = np.empty((store.shape[0], len(anchors), 6), dtype=np.float32)
    model.eval()
    arm = "real" if cfg["arm"] == "sham" else cfg["arm"]
    # Group by anchor to avoid padding unrelated window lengths at inference.
    with torch.inference_mode():
        for column, anchor in enumerate(anchors):
            for begin in range(0, store.shape[0], cfg["batch_size"]):
                people = range(begin, min(store.shape[0], begin + cfg["batch_size"]))
                pairs = [(query, int(anchor)) for query in people]
                batch = _batch(store, pairs, cfg, runtime, permutation)
                logits = model(batch, arm=arm, chunked=True)
                require(bool(torch.isfinite(logits).all()), "nonfinite evaluation logits")
                probabilities[begin:begin + len(pairs), column] = logits.softmax(-1).cpu().numpy()
                _limits(cfg, runtime, started)
    return probabilities


def run_case(train_path: Path, select_path: Path, development_path: Path,
             config_path: Path, outdir: Path) -> dict:
    started = time.monotonic()
    require(not outdir.exists() and not outdir.is_symlink(), "training output already exists")
    cfg = load_config(config_path)
    config_hash = sha256(config_path)
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    train = OrderedContextStore.open(train_path, expected_manifest_sha256=cfg["train_manifest_sha256"])
    select = OrderedContextStore.open(select_path, expected_manifest_sha256=cfg["select_manifest_sha256"])
    data = bind_development(development_path, cfg["development_sha256"], train, select)
    anchors = fixed_anchor_subset(cfg["anchor_count"], train.shape[1], cfg["anchor_seed"])
    labels = data["train"]["truth_state"]
    selected_labels = data["select"]["truth_state"][:, anchors]
    permutation = reference_link_permutation(train, cfg["sham_seed"]) if cfg["arm"] == "sham" else None
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    runtime = DeviceRuntime(cfg["device"], cfg["max_device_bytes"])
    device_info = runtime.configure()
    # CPU initialization makes paired arms start from identical parameter bytes.
    model = OrderedLAIModel(OrderedModelConfig(**cfg["model"]))
    initial = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial.update(name.encode())
        initial.update(value.numpy().tobytes())
    model.to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"], foreach=False, fused=False)
    outdir.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = {"config_sha256": config_hash, "code_sha256": code_hashes,
              "train_manifest_sha256": cfg["train_manifest_sha256"],
              "select_manifest_sha256": cfg["select_manifest_sha256"],
              "development_sha256": cfg["development_sha256"]}
    write_exclusive_json(outdir / "started.json", {"config": cfg, "sources": source,
                         "anchor_indices": anchors.tolist(), "initial_state_sha256": initial.hexdigest()})
    permutation_hash = None
    if permutation is not None:
        permutation_path = outdir / "sham.reference-map.npz"
        write_deterministic_npz(permutation_path, {"reference_person_index": permutation,
                                  "locus_id": train.arrays["locus_id"],
                                  "reference_sample_key_sha256": train.arrays["reference_sample_key_sha256"]})
        permutation_hash = sha256(permutation_path)
    curve, best, best_key = [], None, None
    observations = window_observations = 0
    weighted_loss = train_seconds = eval_seconds = 0.0
    pair_digest = hashlib.sha256()
    exposed_queries, exposed_anchors = set(), set()
    arm = "real" if cfg["arm"] == "sham" else cfg["arm"]
    generator = paired_batches(train.shape[0], anchors, cfg["batch_size"], cfg["steps"], cfg["pair_seed"])
    for step, pairs in enumerate(generator, 1):
        model.train()
        tick = runtime.tick()
        batch = _batch(train, pairs, cfg, runtime, permutation)
        truth = torch.as_tensor([int(labels[q, j]) for q, j in pairs], dtype=torch.long,
                                device=runtime.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, arm=arm, chunked=True)
        loss = F.cross_entropy(logits, truth)
        require(bool(torch.isfinite(loss)), "nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        train_seconds += runtime.elapsed(tick)
        weighted_loss += float(loss.detach().cpu()) * len(pairs)
        observations += len(pairs)
        window_observations += len(pairs)
        pair_digest.update(np.asarray(pairs, dtype="<i8").tobytes())
        exposed_queries.update(query for query, _ in pairs)
        exposed_anchors.update(anchor for _, anchor in pairs)
        _limits(cfg, runtime, started)
        if step % cfg["evaluate_every_steps"] == 0 or step == cfg["steps"]:
            tick = runtime.tick()
            p = predict(model, select, anchors, cfg, runtime, permutation, started=started)
            evaluated = metrics(p, selected_labels)
            eval_seconds += runtime.elapsed(tick)
            row = {"step": step, "train_observations_cumulative": observations,
                   "train_cross_entropy_window": weighted_loss / window_observations,
                   "gradient_norm_last_step": float(norm.detach().cpu()), "SELECT": evaluated,
                   "training_seconds_cumulative": train_seconds, "evaluation_seconds_cumulative": eval_seconds}
            curve.append(row)
            write_exclusive_json(outdir / f"curve-step-{step:07d}.json", row)
            key = (evaluated["brier"], evaluated["log_loss"], step)
            if best_key is None or key < best_key:
                best_key = key
                best = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "step": step, "probabilities": p.copy(), "metrics": evaluated}
            weighted_loss, window_observations = 0.0, 0
            print(json.dumps({"case": cfg["case_id"], "arm": cfg["arm"], "step": step,
                              "SELECT_brier": evaluated["brier"]}), flush=True)
    checkpoint = outdir / "checkpoint.pt"
    with checkpoint.open("xb") as stream:
        torch.save({"state_dict": best["state_dict"], "config": cfg, "selected_step": best["step"],
                    "sources": source, "anchor_indices": anchors.tolist()}, stream)
    predictions = outdir / "select.predictions.npz"
    write_deterministic_npz(predictions, {"probabilities": best["probabilities"],
                            "truth_state": selected_labels, "anchor_indices": anchors,
                            "locus_id": train.arrays["locus_id"][anchors],
                            "sample_key_sha256": data["select"]["sample_key_sha256"]})
    require(sha256(config_path) == config_hash and sha256(development_path) == cfg["development_sha256"]
            and all(sha256(Path(__file__).with_name(k)) == v for k, v in code_hashes.items()),
            "source/config/binding changed during training")
    # Reopen source stores: any modified array aborts before a completion receipt.
    OrderedContextStore.open(train_path, expected_manifest_sha256=cfg["train_manifest_sha256"])
    OrderedContextStore.open(select_path, expected_manifest_sha256=cfg["select_manifest_sha256"])
    with np.load(predictions, allow_pickle=False) as z:
        require(metrics(z["probabilities"], z["truth_state"]) == best["metrics"],
                "saved prediction metrics differ")
    receipt = {"schema_version": SCHEMA, "decision": "COMPLETED_EXPLORATORY_DEVELOPMENT_CASE",
               "config": cfg, "sources": source, "initial_state_sha256": initial.hexdigest(),
               "training_pair_stream_sha256": pair_digest.hexdigest(), "selected_step": best["step"],
               "batch_policy": "interleave_shuffled_anchors_one_same_anchor_minibatch_per_round_remainders_retained",
               "sham_reference_map_sha256": permutation_hash,
               "selected_SELECT_metrics": best["metrics"], "curve": curve,
               "Fminus_SELECT": metrics(data["select"]["baseline"][:, anchors], selected_labels),
               "Ffull_SELECT": metrics(data["select"]["full_baseline"][:, anchors], selected_labels),
               "anchor_indices": anchors.tolist(), "available_anchors": train.shape[1],
               "training_observations": observations,
               "distinct_train_queries_exposed": len(exposed_queries),
               "distinct_train_anchors_exposed": len(exposed_anchors),
               "equivalent_passes_over_selected_train_pairs": observations / (train.shape[0] * len(anchors)),
               "training_seconds": train_seconds, "evaluation_seconds": eval_seconds,
               "elapsed_seconds": time.monotonic() - started, "device": device_info,
               "gpu_memory": runtime.memory(), "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "checkpoint_sha256": sha256(checkpoint), "predictions_sha256": sha256(predictions),
               "selection": "minimum_SELECT_Brier_then_log_loss_then_earliest_step",
               "normalization": "fixed_binary_channels_and_delta_cm_divided_by_declared_radius",
               "scope": {"SCORE_opened": False, "source_TEST_or_VALID_opened": False,
                         "rare_phase_assigned": False, "dense_LAI_or_border_F1_evaluated": False,
                         "convergence_demonstrated": False, "new_cloud_instances_created_by_runner": 0,
                         "SHAM_is_exact_permutation_test": False}}
    write_exclusive_json(outdir / "training.receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("train-store", "select-store", "development", "config", "outdir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    run_case(args.train_store, args.select_store, args.development, args.config, args.outdir)


if __name__ == "__main__":
    main()
