#!/usr/bin/env python3
"""Seal SCORE-only predictions from one trained M33 DEVELOPMENT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m33_t0a_models as models
import m33_train_development as train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--rotation", choices=("R0", "R1", "R2"), required=True)
    parser.add_argument("--family", choices=models.FAMILIES, required=True)
    parser.add_argument("--radius", type=float, choices=train.materialize.RADII, required=True)
    parser.add_argument("--arm", choices=("RD", "RE"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    os.environ.setdefault("USER", "m33")
    os.environ.setdefault("LOGNAME", "m33")
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    spec = next(row for row in pre4["root_registry"]["development_rotations"]
                if row["rotation"] == args.rotation)
    score_seed = int(spec["score_only_root"])
    root = train.load_root(args.runtime, score_seed, False)
    norm_path = args.runtime / "materialized" / args.rotation / "fit_callable_normalization_manifest.json"
    norm = json.loads(norm_path.read_text(encoding="utf-8"))["max_callable_an"]
    norm_sha = train.sha256_file(norm_path)
    model = models.build_model(args.family)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    probabilities = train.predict_root(
        model, root, root.target, root.f0, root.reference, norm,
        args.rotation, norm_sha, args.arm, args.radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train.write_prediction(args.output, root, root.target, probabilities)
    receipt = {
        "schema_version": "1.0.0", "stage": "M33_DEVELOPMENT_PREDICT",
        "status": "PASS_SCORE_PREDICTION_SEALED", "rotation": args.rotation,
        "score_only_root": score_seed, "family": args.family, "radius_cM": args.radius,
        "arm": args.arm, "checkpoint_sha256": train.sha256_file(args.checkpoint),
        "prediction_sha256": train.sha256_file(args.output),
        "score_truth_argument_available": False, "score_truth_accessed": False,
        "wall_seconds": time.monotonic() - started,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return receipt


if __name__ == "__main__":
    run(parse_args())
