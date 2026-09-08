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
                                      reference_link_permutation, require, sham_dose_changes,
                                      stratified_metrics)
from m39_profile_device import DeviceRuntime

SCHEMA = "m39-ordered-anchor-training-v1"
ARMS = ("common", "pooled", "real", "sham")
SOURCE_FILES = ("m39_ordered_training.py", "m39_ordered_training_data.py", "m39_ordered_models.py",
                "m39_ordered_batches.py", "m39_ordered_context.py", "m39_profile_device.py",
                "m39_anchor_screen.py", "m39_carrier_models.py", "m33_safe_bridge_core.py",
                "m34_prepare_panel_factors.py", "m34_generate_mosaics.py")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    return validate_config(cfg)


def validate_config(cfg: dict, *, schema: str = SCHEMA, arms=ARMS,
                    extra_required=frozenset()) -> dict:
    """Shared run contract; alternate representations declare their own schema."""
    required = {"schema_version", "case_id", "scope", "arm", "model", "seed", "pair_seed",
                "anchor_seed", "sham_seed", "anchor_count", "steps", "evaluate_every_steps",
                "batch_size", "learning_rate", "weight_decay", "gradient_clip_norm",
                "device", "cpu_threads", "max_input_bytes", "max_device_bytes", "max_rss_bytes",
                "max_runtime_seconds", "train_manifest_sha256", "select_manifest_sha256",
                "development_sha256", "selection_metric", "paired_budget_id"}
    optional = {"train_probe_people", "evaluate_initial"}
    required |= set(extra_required)
    require(required <= set(cfg) <= required | optional and cfg["schema_version"] == schema,
            "training config fields differ")
    require(type(cfg.get("train_probe_people", 0)) is int and cfg.get("train_probe_people", 0) >= 0,
            "invalid TRAIN probe size")
    require(type(cfg.get("evaluate_initial", False)) is bool, "invalid initial evaluation flag")
    require(cfg["scope"] == "exploratory_chr22_R0_development_anchors_only"
            and cfg["arm"] in arms and cfg["selection_metric"] == "brier", "training scope/arm differs")
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


class OrderedTrainingBackend:
    """Representation hooks; the historical optimizer, schedule and math stay shared."""

    schema = SCHEMA
    source_files = SOURCE_FILES
    persist_intermediate_best = False

    def load_config(self, path):
        return load_config(path)

    def prepare(self, train, select, data, cfg):
        pass

    def make_model(self, cfg):
        return OrderedLAIModel(OrderedModelConfig(**cfg["model"]))

    def make_batch(self, store, pairs, cfg, runtime, permutation):
        return _batch(store, pairs, cfg, runtime, permutation)

    def probabilities(self, model, batch, cfg):
        arm = "real" if cfg["arm"] == "sham" else cfg["arm"]
        logits = model(batch, arm=arm, chunked=True)
        require(bool(torch.isfinite(logits).all()), "nonfinite evaluation logits")
        return logits.softmax(-1)

    def loss(self, model, batch, truth, cfg):
        arm = "real" if cfg["arm"] == "sham" else cfg["arm"]
        return F.cross_entropy(model(batch, arm=arm, chunked=True), truth)

    def diagnostics(self, model, cfg):
        return {}

    def checkpoint_eligible(self, step):
        return True


def predict(model, store, anchors, cfg, runtime, permutation, *, started, people=None, backend=None):
    """Evaluate declared anchors; an explicit TRAIN probe may subset people, SELECT never does."""
    people = np.arange(store.shape[0], dtype=np.int64) if people is None else np.asarray(people)
    require(people.ndim == 1 and people.dtype.kind in "iu" and len(people) > 0
            and len(np.unique(people)) == len(people)
            and int(people.min()) >= 0 and int(people.max()) < store.shape[0],
            "invalid evaluation people")
    probabilities = np.empty((len(people), len(anchors), 6), dtype=np.float32)
    model.eval()
    backend = backend or OrderedTrainingBackend()
    # Group by anchor to avoid padding unrelated window lengths at inference.
    with torch.inference_mode():
        for column, anchor in enumerate(anchors):
            for begin in range(0, len(people), cfg["batch_size"]):
                queries = people[begin:begin + cfg["batch_size"]]
                pairs = [(int(query), int(anchor)) for query in queries]
                batch = backend.make_batch(store, pairs, cfg, runtime, permutation)
                probabilities[begin:begin + len(pairs), column] = backend.probabilities(model, batch, cfg).cpu().numpy()
                _limits(cfg, runtime, started)
    return probabilities


def exposure_metrics(visits: np.ndarray) -> dict:
    """Count exposure, not independent biological observations or convergence."""
    require(visits.ndim == 2 and visits.dtype.kind in "iu" and visits.size > 0,
            "invalid visit counter")
    total = int(visits.sum())
    return {"observations": total, "unique_pairs": int(np.count_nonzero(visits)),
            "available_pairs": int(visits.size),
            "unique_pair_fraction": float(np.count_nonzero(visits) / visits.size),
            "distinct_people": int(np.count_nonzero(np.any(visits > 0, axis=1))),
            "distinct_anchors": int(np.count_nonzero(np.any(visits > 0, axis=0))),
            "minimum_visits_per_pair": int(visits.min()),
            "maximum_visits_per_pair": int(visits.max()),
            "complete_passes_over_declared_pairs": int(visits.min()),
            "equivalent_passes_over_declared_pairs": total / visits.size}


def run_case(train_path: Path, select_path: Path, development_path: Path,
             config_path: Path, outdir: Path, *, backend=None) -> dict:
    started = time.monotonic()
    require(not outdir.exists() and not outdir.is_symlink(), "training output already exists")
    backend = backend or OrderedTrainingBackend()
    cfg = backend.load_config(config_path)
    config_hash = sha256(config_path)
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in backend.source_files}
    train = OrderedContextStore.open(train_path, expected_manifest_sha256=cfg["train_manifest_sha256"])
    select = OrderedContextStore.open(select_path, expected_manifest_sha256=cfg["select_manifest_sha256"])
    data = bind_development(development_path, cfg["development_sha256"], train, select)
    backend.prepare(train, select, data, cfg)
    anchors = fixed_anchor_subset(cfg["anchor_count"], train.shape[1], cfg["anchor_seed"])
    probe_count = cfg.get("train_probe_people", 0)
    require(probe_count <= train.shape[0], "TRAIN probe exceeds role size")
    # Draw the probe before reading labels. It is identical across paired arms and
    # used for learning diagnostics only, never for choosing a checkpoint.
    probe_people = (fixed_anchor_subset(probe_count, train.shape[0], cfg["pair_seed"])
                    if probe_count else None)
    labels = data["train"]["truth_state"]
    selected_labels = data["select"]["truth_state"][:, anchors]
    permutation = reference_link_permutation(train, cfg["sham_seed"]) if cfg["arm"] == "sham" else None
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    runtime = DeviceRuntime(cfg["device"], cfg["max_device_bytes"])
    device_info = runtime.configure()
    # CPU initialization makes paired arms start from identical parameter bytes.
    model = backend.make_model(cfg)
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
    model_diagnostics = backend.diagnostics(model, cfg)
    if model_diagnostics:
        source["representation"] = model_diagnostics
    write_exclusive_json(outdir / "started.json", {"config": cfg, "sources": source,
                         "anchor_indices": anchors.tolist(), "initial_state_sha256": initial.hexdigest()})
    permutation_hash = None
    sham_diagnostic = None
    if permutation is not None:
        permutation_path = outdir / "sham.reference-map.npz"
        write_deterministic_npz(permutation_path, {"reference_person_index": permutation,
                                  "locus_id": train.arrays["locus_id"],
                                  "reference_sample_key_sha256": train.arrays["reference_sample_key_sha256"]})
        permutation_hash = sha256(permutation_path)
        sham_diagnostic = sham_dose_changes(train, anchors, permutation)
        write_exclusive_json(outdir / 'sham.design-diagnostic.json', sham_diagnostic)
    curve, best, best_key = [], None, None
    observations = window_observations = 0
    weighted_loss = train_seconds = eval_seconds = 0.0
    select_seconds = probe_seconds = 0.0
    pair_digest = hashlib.sha256()
    exposed_queries, exposed_anchors = set(), set()
    visits = np.zeros((train.shape[0], len(anchors)), dtype=np.int64)
    anchor_columns = {int(anchor): column for column, anchor in enumerate(anchors)}

    def evaluate_checkpoint(step, online_loss, gradient_norm):
        nonlocal eval_seconds, select_seconds, probe_seconds, best_key, best
        tick = runtime.tick()
        p = predict(model, select, anchors, cfg, runtime, permutation, started=started, backend=backend)
        evaluated = metrics(p, selected_labels)
        select_seconds += runtime.elapsed(tick)
        probe_metrics = None
        if probe_people is not None:
            tick = runtime.tick()
            probe = predict(model, train, anchors, cfg, runtime, permutation,
                            started=started, people=probe_people, backend=backend)
            probe_metrics = metrics(probe, labels[probe_people][:, anchors])
            probe_seconds += runtime.elapsed(tick)
        eval_seconds = select_seconds + probe_seconds
        row = {"step": step, "train_observations_cumulative": observations,
               "train_cross_entropy_window": online_loss,
               "train_cross_entropy_semantics": "online_before_each_update_not_frozen_checkpoint_loss",
               "gradient_norm_last_step": gradient_norm, "SELECT": evaluated,
               "TRAIN_probe": probe_metrics, "TRAIN_exposure": exposure_metrics(visits),
               "training_seconds_cumulative": train_seconds,
               "evaluation_seconds_cumulative": eval_seconds,
               "SELECT_seconds_cumulative": select_seconds,
               "TRAIN_probe_seconds_cumulative": probe_seconds}
        if backend.persist_intermediate_best:
            row["checkpoint_eligible"] = backend.checkpoint_eligible(step)
        curve.append(row)
        write_exclusive_json(outdir / f"curve-step-{step:07d}.json", row)
        key = (evaluated["brier"], evaluated["log_loss"], step)
        if backend.checkpoint_eligible(step) and (best_key is None or key < best_key):
            best_key = key
            best = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "step": step, "probabilities": p.copy(), "metrics": evaluated}
            if backend.persist_intermediate_best:
                # Immutable recovery points survive a later budget failure; they
                # are not a completion receipt or a license to compare unequal runs.
                with (outdir / f"checkpoint-step-{step:07d}.pt").open("xb") as stream:
                    torch.save({"state_dict": best["state_dict"], "config": cfg,
                                "selected_step": step, "sources": source,
                                "anchor_indices": anchors.tolist()}, stream)
                write_deterministic_npz(outdir / f"select-step-{step:07d}.npz",
                                        {"probabilities": p, "anchor_indices": anchors,
                                         "sample_key_sha256": data["select"]["sample_key_sha256"]})
        print(json.dumps({"case": cfg["case_id"], "arm": cfg["arm"], "step": step,
                          "SELECT_brier": evaluated["brier"],
                          "complete_passes": int(visits.min())}), flush=True)

    if cfg.get("evaluate_initial", False):
        evaluate_checkpoint(0, None, None)
    generator = paired_batches(train.shape[0], anchors, cfg["batch_size"], cfg["steps"], cfg["pair_seed"])
    for step, pairs in enumerate(generator, 1):
        model.train()
        tick = runtime.tick()
        batch = backend.make_batch(train, pairs, cfg, runtime, permutation)
        truth = torch.as_tensor([int(labels[q, j]) for q, j in pairs], dtype=torch.long,
                                device=runtime.device)
        optimizer.zero_grad(set_to_none=True)
        loss = backend.loss(model, batch, truth, cfg)
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
        for query, anchor in pairs:
            visits[query, anchor_columns[anchor]] += 1
        _limits(cfg, runtime, started)
        if step % cfg["evaluate_every_steps"] == 0 or step == cfg["steps"]:
            evaluate_checkpoint(step, weighted_loss / window_observations, float(norm.detach().cpu()))
            weighted_loss, window_observations = 0.0, 0
    checkpoint = outdir / "checkpoint.pt"
    with checkpoint.open("xb") as stream:
        torch.save({"state_dict": best["state_dict"], "config": cfg, "selected_step": best["step"],
                    "sources": source, "anchor_indices": anchors.tolist()}, stream)
    predictions = outdir / "select.predictions.npz"
    observed = select.arrays['query_observed'][:, anchors].astype(bool)
    carrier = observed & (select.arrays['query_dosage'][:, anchors] > 0)
    stratified = {name: stratified_metrics(p, selected_labels, carrier, observed) for name, p in (
        ('model', best['probabilities']), ('Fminus', data['select']['baseline'][:, anchors]),
        ('Ffull', data['select']['full_baseline'][:, anchors]))}
    write_deterministic_npz(predictions, {"probabilities": best["probabilities"],
                            "truth_state": selected_labels, "anchor_indices": anchors,
                            "locus_id": train.arrays["locus_id"][anchors],
                            "query_carrier": carrier, "query_observed": observed,
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
    receipt = {"schema_version": backend.schema, "decision": "COMPLETED_EXPLORATORY_DEVELOPMENT_CASE",
               "config": cfg, "sources": source, "initial_state_sha256": initial.hexdigest(),
               "training_pair_stream_sha256": pair_digest.hexdigest(), "selected_step": best["step"],
               "batch_policy": "interleave_shuffled_anchors_one_same_anchor_minibatch_per_round_remainders_retained",
               "sham_reference_map_sha256": permutation_hash,
               "sham_TRAIN_dose_changes": sham_diagnostic,
               "selected_SELECT_metrics": best["metrics"], "curve": curve,
               "selected_SELECT_stratified_descriptive_only": stratified,
               "Fminus_SELECT": metrics(data["select"]["baseline"][:, anchors], selected_labels),
               "Ffull_SELECT": metrics(data["select"]["full_baseline"][:, anchors], selected_labels),
               "anchor_indices": anchors.tolist(), "available_anchors": train.shape[1],
               "training_observations": observations,
               "distinct_train_queries_exposed": len(exposed_queries),
               "distinct_train_anchors_exposed": len(exposed_anchors),
               "equivalent_passes_over_selected_train_pairs": observations / (train.shape[0] * len(anchors)),
               "training_seconds": train_seconds, "evaluation_seconds": eval_seconds,
               "SELECT_seconds": select_seconds, "TRAIN_probe_seconds": probe_seconds,
               "TRAIN_exposure": exposure_metrics(visits),
               "TRAIN_probe_people": probe_people.tolist() if probe_people is not None else [],
               "budget_diagnostics": {"best_checkpoint_is_initial": best["step"] == 0,
                    "best_checkpoint_at_budget_end": best["step"] == cfg["steps"],
                    "last_SELECT_brier_improved_over_previous":
                        curve[-1]["SELECT"]["brier"] < curve[-2]["SELECT"]["brier"] if len(curve) > 1 else None,
                    "negative_family_conclusion_allowed": False,
                    "reason": "fixed_budget_single_development_split_does_not_establish_convergence"},
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
