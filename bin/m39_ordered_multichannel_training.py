#!/usr/bin/env python3
"""Bounded, separately fitted multichannel anchor ablations over TRAIN/SELECT.

The historical ordered trainer owns scheduling, optimizer, provenance, limits and
checkpoint selection. This backend changes only inputs, model and its native
log-probability loss. OFF is the authenticated Fminus comparator, never a fit.
No dense interpolation, SCORE access or cloud launch is implemented here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F

import m39_ordered_training as training
from m39_ordered_models import OrderedLAIModel, OrderedModelConfig
from m39_ordered_multichannel import MultichannelConfig, OrderedMultichannelAdapter
from m39_ordered_multichannel_data import ReferenceSummary, pack_multichannel_batch
from m39_ordered_training_data import require


SCHEMA = "m39-ordered-multichannel-training-v1"
ARMS = ("none", "summary", "detail", "both", "sham")
SOURCE_FILES = training.SOURCE_FILES + (
    "m39_ordered_multichannel.py", "m39_ordered_multichannel_data.py",
    "m39_ordered_multichannel_training.py",
)


def load_config(path: Path) -> dict:
    cfg = training.validate_config(json.loads(path.read_text()), schema=SCHEMA,
                                   arms=ARMS, extra_required={"multichannel"})
    require(isinstance(cfg["multichannel"], dict), "multichannel configuration must be a mapping")
    require(set(cfg["multichannel"]) == {"hidden_width", "initial_gate"},
            "all multichannel architecture values must be explicit")
    MultichannelConfig(**cfg["multichannel"])
    return cfg


def model_mode(arm: str) -> str:
    require(arm in ARMS, "unknown trainable multichannel arm")
    return "BOTH" if arm == "sham" else arm.upper()


class MultichannelTrainingBackend(training.OrderedTrainingBackend):
    schema = SCHEMA
    source_files = SOURCE_FILES
    persist_intermediate_best = True

    def load_config(self, path):
        return load_config(path)

    def prepare(self, train, select, data, cfg):
        # bind_development already authenticated identical REF arrays and role
        # axes. This cache contains counts only, never target truth or outcomes.
        self.summary = ReferenceSummary.from_store(train)
        self.baselines = {id(train): data["train"]["baseline"],
                          id(select): data["select"]["baseline"]}

    def make_model(self, cfg):
        # Always initialize the complete inventory before disabling unused
        # branches so paired arms retain exactly the same initial state bytes.
        model = OrderedMultichannelAdapter(OrderedLAIModel(OrderedModelConfig(**cfg["model"])),
                                          MultichannelConfig(**cfg["multichannel"]))
        inactive = [model.encoder.head]
        mode = model_mode(cfg["arm"])
        if mode not in ("SUMMARY", "BOTH"):
            inactive.append(model.summary_projection)
        if mode not in ("DETAIL", "BOTH"):
            inactive.extend((model.detail_candidate, model.detail_projection))
        for branch in inactive:
            branch.requires_grad_(False)
        return model

    def make_batch(self, store, pairs, cfg, runtime, permutation):
        require(id(store) in self.baselines, "multichannel role has not been authenticated")
        batch = pack_multichannel_batch(store, pairs, self.baselines[id(store)], self.summary,
                                       mode=cfg["arm"].upper(),
                                       max_input_bytes=cfg["max_input_bytes"],
                                       sham_permutation=permutation)
        return runtime.transfer(batch)

    @staticmethod
    def forward(model, batch, cfg):
        return model(batch, batch["baseline"], batch, mode=model_mode(cfg["arm"]), chunked=True)

    def probabilities(self, model, batch, cfg):
        probabilities = self.forward(model, batch, cfg)["probabilities"]
        require(bool(torch.isfinite(probabilities).all()), "nonfinite multichannel probabilities")
        # Do not softmax log probabilities again: preserve the adapter's exact
        # baseline fallback and the small representable mixture mass near g=1.
        return probabilities

    def loss(self, model, batch, truth, cfg):
        return F.nll_loss(self.forward(model, batch, cfg)["log_probabilities"], truth)

    def checkpoint_eligible(self, step):
        # Initial metrics diagnose optimization but cannot select an untrained
        # model. Scientific sweep orchestration freezes completed-pass intervals.
        return step > 0

    def diagnostics(self, model, cfg):
        parameters = dict(model.named_parameters())
        return {"kind": "ordered_multichannel_FLARE_mixture", "mode": model_mode(cfg["arm"]),
                "multichannel": asdict(model.config),
                "parameter_count": sum(p.numel() for p in parameters.values()),
                "enabled_parameter_count": sum(p.numel() for p in parameters.values() if p.requires_grad),
                "disabled_parameter_names": [name for name, p in parameters.items() if not p.requires_grad],
                "enabled_count_is_not_effective_degrees_of_freedom": True,
                "baseline": "authenticated_Fminus_detached_no_gradient",
                "OFF": "exact_Fminus_not_separately_trained",
                "loss": "negative_native_log_mixture_probability_no_second_softmax_no_added_floor",
                "initial_checkpoint_selectable": False,
                "SUMMARY_universe": "all_unique_eligible_REF_people_per_locus_and_ancestry",
                "DETAIL_universe": "common_retrieved_REF_candidates_diploid_dosage",
                "SUMMARY_and_DETAIL_have_different_available_reference_universes": True}


def run_case(train_path: Path, select_path: Path, development_path: Path,
             config_path: Path, outdir: Path) -> dict:
    return training.run_case(train_path, select_path, development_path, config_path, outdir,
                             backend=MultichannelTrainingBackend())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("train-store", "select-store", "development", "config", "outdir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    run_case(args.train_store, args.select_store, args.development, args.config, args.outdir)


if __name__ == "__main__":
    main()
