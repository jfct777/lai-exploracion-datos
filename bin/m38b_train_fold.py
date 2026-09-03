#!/usr/bin/env python3
"""Fit one M38B candidate without access to SCORE truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import TraceSpec, require
from m37_trace_train import deterministic_train_tune, load_features, load_truth, train
from m38b_oof_core import (
    analytic_residual,
    per_person_log_loss,
    smooth_evidence_triangular,
)
from m38b_score_oof import normalised_voronoi_cm_weights


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_model_contract_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_AUTHENTICATE_MODEL_CONTRACT"
        and receipt.get("status") == "PASS_BASE_AND_PRE_OUTCOME_AMENDMENT_BOUND"
        and receipt.get("arm_binding_required") is True
        and receipt.get("scope", {}).get("partition") == "FIT"
        and receipt.get("scope", {}).get("people") == 96
        and receipt.get("scope", {}).get("valid_opened") is False
        and receipt.get("scope", {}).get("test_opened") is False
        and receipt.get("source_binding") == "DETERMINISTIC_LOAD_BEARING_SOURCE_MANIFEST"
        and isinstance(receipt.get("source_manifest_sha256"), str)
        and len(receipt["source_manifest_sha256"]) == 64
        and len(receipt.get("source_manifest", [])) == 27,
        "M38B authenticated model contract differs",
    )
    return receipt


def verify_partition_receipt(
    receipt_path: Path, fit_features: Path, score_features: Path, fold: int,
    expected_arm: str,
) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_PARTITION_FEATURES"
        and receipt.get("status") == "PASS_TRUTH_BLIND_FEATURE_PARTITION"
        and receipt.get("fold") == fold
        and receipt.get("arm") == expected_arm
        and receipt.get("source_arm") == expected_arm
        and receipt.get("truth_read") is False,
        "M38B feature partition receipt differs",
    )
    require(
        receipt.get("fit_output_sha256") == sha256(fit_features)
        and receipt.get("score_output_sha256") == sha256(score_features),
        "M38B feature partition hashes differ",
    )
    if expected_arm == "POSITIVE":
        require(
            receipt.get("diagnostic_only") is True
            and receipt.get("source_stage") == "M38B_POSITIVE_CONTROL_MATERIALIZE"
            and receipt.get("positive_control_delta") in {0.0, 0.25, 0.5, 1.0, 2.0},
            "M38B POSITIVE partition provenance differs",
        )
        require(all(isinstance(receipt.get(name), str) and len(receipt[name]) == 64
                    for name in ("real_event_identity_sha256", "real_event_masks_sha256")),
                "M38B POSITIVE event identity provenance differs")
    else:
        require(
            receipt.get("diagnostic_only") is False
            and receipt.get("source_stage") == "M37_TRACE_MATERIALIZE"
            and receipt.get("positive_control_delta") is None,
            "M38B production partition provenance differs",
        )
    return receipt


def verify_fit_truth(
    truth_path: Path, fit: dict[str, np.ndarray], receipt_path: Path, fold: int,
) -> np.ndarray:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_PARTITION_TRUTH"
        and receipt.get("status") == "PASS_NON_SELECTING_TRUTH_PARTITION"
        and receipt.get("fold") == fold
        and receipt.get("fit_output_sha256") == sha256(truth_path),
        "M38B FIT truth receipt differs",
    )
    with np.load(truth_path, allow_pickle=False) as archive:
        require(
            {"sample_key_sha256", "marker_pos", "state_labels"}.issubset(archive.files)
            and np.array_equal(archive["sample_key_sha256"], fit["sample_key_sha256"])
            and np.array_equal(archive["marker_pos"], fit["marker_pos"]),
            "M38B FIT truth/features axes differ",
        )
    return load_truth(truth_path)


def analytic_prediction(
    fit: dict[str, np.ndarray], score: dict[str, np.ndarray], truth: np.ndarray,
    inner_seed: int, radius_cm: float, lambda_grid: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    _train_people, select_people = deterministic_train_tune(fit, inner_seed, 0.25)
    require(len(select_people) == 16, "M38B analytical SELECT size differs")
    fit_field = smooth_evidence_triangular(fit["evidence_field"], fit["marker_cM"], radius_cm)
    score_field = smooth_evidence_triangular(score["evidence_field"], score["marker_cM"], radius_cm)
    rows: list[dict[str, float]] = []
    for strength in lambda_grid:
        probability = analytic_residual(fit["baseline_states"], fit_field, strength)
        per_person = per_person_log_loss(
            probability[select_people], truth[select_people], fit["marker_cM"], weighted=True,
        )
        rows.append({"lambda": float(strength), "select_log_loss": float(per_person.mean())})
    selected = min(rows, key=lambda row: (row["select_log_loss"], row["lambda"]))
    prediction = analytic_residual(
        score["baseline_states"], score_field, float(selected["lambda"]),
    )
    return prediction, {"grid": rows, "selected_lambda": selected["lambda"],
                        "selection_people": int(len(select_people))}


def run(args: argparse.Namespace) -> dict[str, object]:
    require(args.arm in {"RE", "RD", "SHAM", "POSITIVE"}, "M38B candidate arm differs")
    require(args.family in {"analytic", "tcn"}, "M38B candidate family differs")
    require(args.arm != "RD",
            "M38B RD/OFF is exact F-minus-S660 and must not be fitted")
    model_contract = verify_model_contract_receipt(args.model_contract_receipt)
    feature_receipt = verify_partition_receipt(
        args.feature_receipt, args.fit_features, args.score_features, args.fold,
        args.arm,
    )
    if args.arm == "POSITIVE":
        require(feature_receipt.get("diagnostic_only") is True,
                "M38B POSITIVE training needs diagnostic-only provenance")
        positive_delta = float(feature_receipt["positive_control_delta"])
    else:
        positive_delta = None
    fit, score = load_features(args.fit_features), load_features(args.score_features)
    require(
        np.array_equal(fit["marker_pos"], score["marker_pos"])
        and np.array_equal(fit["marker_cM"], score["marker_cM"])
        and len(np.intersect1d(fit["sample_key_sha256"], score["sample_key_sha256"])) == 0,
        "M38B FIT/SCORE feature axes differ",
    )
    truth = verify_fit_truth(args.fit_truth, fit, args.truth_receipt, args.fold)
    inner_seed = int(feature_receipt["inner_split_seed"])
    train_people, select_people = deterministic_train_tune(fit, inner_seed, 0.25)
    require(len(train_people) == 48 and len(select_people) == 16,
            "M38B inner split differs")
    diagnostics: dict[str, object]
    checkpoint: Path | None = None
    if args.family == "analytic":
        grid = tuple(float(value) for value in args.lambda_grid.split(","))
        require(grid == (0.0, 0.25, 0.5, 1.0, 2.0),
                "M38B analytical lambda grid differs")
        prediction, diagnostics = analytic_prediction(
            fit, score, truth, inner_seed, args.event_radius_cm, grid,
        )
    else:
        require(
            args.hidden_dim == 96 and args.depth == 3 and args.kernel_size == 5
            and args.dropout == 0.0 and args.learning_rate == 1e-4
            and args.updates == 800 and args.event_radius_cm == 0.2
            and args.evidence_scale == 2.0
            and args.dilations == "1,2,4"
            and args.batch_people == 8 and args.marker_shard == 256
            and args.validation_every == 25 and args.patience == 4
            and args.seed in {1103, 2207, 3301},
            "M38B TCN differs from the M37 capacity-passing candidate",
        )
        training_diagnostics: dict[str, int | bool | None] = {}
        checkpoint = args.checkpoint
        prediction, observed_select = train(
            fit, score, truth, "tcn", 12.0, args.evidence_scale,
            TraceSpec(args.hidden_dim, args.depth, args.kernel_size, args.dropout,
                      tuple(int(value) for value in args.dilations.split(","))),
            args.updates, args.learning_rate, args.batch_people, args.marker_shard,
            args.validation_every, args.patience, args.seed, checkpoint, 0.25,
            inner_seed, args.event_radius_cm, training_diagnostics,
            loss_marker_weights=normalised_voronoi_cm_weights(fit["marker_cM"]),
            anchor_event_carrier=True,
        )
        require(np.array_equal(observed_select, select_people),
                "M38B trainer SELECT assignment drifted")
        diagnostics = {"training": training_diagnostics, "selection_people": 16}
    require(
        prediction.shape == (32, len(score["marker_pos"]), 6)
        and np.isfinite(prediction).all()
        and np.all(prediction >= 0)
        and np.allclose(prediction.sum(axis=2), 1.0, rtol=0, atol=5e-6),
        "M38B SCORE prediction differs",
    )
    payload = {
        "probabilities": np.ascontiguousarray(prediction, dtype=np.float32),
        "sample_key_sha256": score["sample_key_sha256"],
        "marker_pos": score["marker_pos"],
        "marker_cM": score["marker_cM"],
        "marker_axis_sha256": score["marker_axis_sha256"],
        "fold": np.asarray([args.fold], dtype=np.uint8),
        "family": np.asarray([args.family]),
        "arm": np.asarray([args.arm]),
        "seed": np.asarray([args.seed], dtype=np.int64),
    }
    if positive_delta is not None:
        payload["positive_delta"] = np.asarray([positive_delta], dtype=np.float64)
    require(not args.output.exists(), "refusing to overwrite M38B prediction")
    write_deterministic_npz(args.output, payload)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M38B_TRAIN_AND_PREDICT_OOF",
        "status": ("PASS_DIAGNOSTIC_TRUTH_ENCODED_POSITIVE_CONTROL"
                   if args.arm == "POSITIVE" else "PASS_SCORE_TRUTH_INACCESSIBLE"),
        "fold": int(args.fold),
        "family": args.family,
        "arm": args.arm,
        "seed": int(args.seed),
        "positive_control_delta": positive_delta,
        "real_event_identity_sha256": feature_receipt.get("real_event_identity_sha256"),
        "real_event_masks_sha256": feature_receipt.get("real_event_masks_sha256"),
        "diagnostic_only": args.arm == "POSITIVE",
        "inner_split_seed": inner_seed,
        "train_people": 48,
        "select_people": 16,
        "score_people": 32,
        "score_truth_input": None,
        "fit_features_sha256": sha256(args.fit_features),
        "score_features_sha256": sha256(args.score_features),
        "fit_truth_sha256": sha256(args.fit_truth),
        "feature_receipt_sha256": sha256(args.feature_receipt),
        "truth_receipt_sha256": sha256(args.truth_receipt),
        "model_contract_receipt_sha256": sha256(args.model_contract_receipt),
        "base_contract_sha256": model_contract["base_contract_sha256"],
        "amendment_sha256": model_contract["amendment_sha256"],
        "amendment_2_sha256": model_contract["amendment_2_sha256"],
        "diagnostics": diagnostics,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "output_sha256": sha256(args.output),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("analytic", "tcn"), required=True)
    parser.add_argument("--arm", choices=("RE", "RD", "SHAM", "POSITIVE"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fit-features", type=Path, required=True)
    parser.add_argument("--score-features", type=Path, required=True)
    parser.add_argument("--feature-receipt", type=Path, required=True)
    parser.add_argument("--fit-truth", type=Path, required=True)
    parser.add_argument("--truth-receipt", type=Path, required=True)
    parser.add_argument("--model-contract-receipt", type=Path, required=True)
    parser.add_argument("--lambda-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--event-radius-cm", type=float, default=0.2)
    parser.add_argument("--evidence-scale", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dilations", default="1,2,4")
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-people", type=int, default=8)
    parser.add_argument("--marker-shard", type=int, default=256)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), "refusing to overwrite M38B training receipt")
    print(json.dumps({"status": run(args)["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
