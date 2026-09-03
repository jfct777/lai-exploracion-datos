#!/usr/bin/env python3
"""Assemble truth-blind M38B out-of-fold predictions on the original FIT axis.

Each of the 96 people must occur in SCORE in exactly one outer fold.  For the
TCN, probabilities from the three preregistered seeds are averaged; no seed is
selected using outcomes.  Local-ancestry truth is intentionally not accepted
by this interface and is joined only later by the scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from m33_safe_bridge_core import reopen_npz, write_deterministic_npz, write_exclusive_json


STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")
TCN_SEEDS = (1103, 2207, 3301)
ANALYTIC_SEEDS = (1103,)
PREDICTION_REQUIRED = {
    "probabilities", "sample_key_sha256", "marker_pos", "marker_cM",
    "marker_axis_sha256", "fold", "family", "arm", "seed",
}
HEX_DIGITS = frozenset("0123456789abcdef")


class M38BOofCollectionError(ValueError):
    """Raised when OOF coverage, seed completeness, or an axis differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BOofCollectionError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"input is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_scalar(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    require(array.size == 1, f"{label} must be scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, (bytes, np.bytes_)):
        return bytes(item).decode("ascii")
    return str(item)


def _load_folds(
    path: Path, receipt_path: Path, expected_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    require(
        len(expected_sha256) == 64 and set(expected_sha256).issubset(HEX_DIGITS),
        "fold expected SHA-256 is malformed",
    )
    digest = sha256_file(path)
    require(digest == expected_sha256, "fold SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        require(
            set(archive.files) == {
                "sample_key_sha256", "roles", "outer_fold",
                "inner_split_seed", "outer_seed",
            },
            "fold NPZ members differ",
        )
        samples = np.ascontiguousarray(archive["sample_key_sha256"])
        roles = np.ascontiguousarray(archive["roles"])
        outer_fold = np.ascontiguousarray(archive["outer_fold"])
        inner_seeds = np.ascontiguousarray(archive["inner_split_seed"])
    require(
        samples.dtype == np.dtype("|S64")
        and samples.shape == (96,)
        and np.unique(samples).size == 96
        and roles.shape == (3, 96)
        and np.array_equal(outer_fold, np.arange(3, dtype=np.uint8))
        and inner_seeds.shape == (3,),
        "fold axes or dtypes differ",
    )
    require(
        set(np.unique(roles).tolist()) == {"TRAIN", "SELECT", "SCORE"}
        and all(np.count_nonzero(roles[fold] == "TRAIN") == 48 for fold in range(3))
        and all(np.count_nonzero(roles[fold] == "SELECT") == 16 for fold in range(3))
        and all(np.count_nonzero(roles[fold] == "SCORE") == 32 for fold in range(3))
        and np.all(np.sum(roles == "SCORE", axis=0) == 1),
        "fold role counts or OOF coverage differ",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_FREEZE_OOF_ROTATION"
        and receipt.get("status") == "PASS_TRUTH_BLIND_EXACT_ROTATION"
        and receipt.get("output_sha256") == digest
        and receipt.get("truth_read") is False
        and receipt.get("target_genotypes_read") is False
        and receipt.get("folds") == 3
        and receipt.get("people") == 96
        and receipt.get("score_appearances_per_person") == 1,
        "fold receipt differs",
    )
    return samples, roles, receipt, digest


def _load_prediction(
    path: Path, receipt_path: Path, expected_family: str, expected_arm: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], int, int, str]:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        require(PREDICTION_REQUIRED.issubset(archive.files),
                "prediction NPZ members differ")
        forbidden = {"truth", "labels", "state_labels", "score_truth"}
        require(not forbidden.intersection(archive.files),
                "truth is forbidden in the OOF collector")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    family = decode_scalar(arrays["family"], "prediction family")
    arm = decode_scalar(arrays["arm"], "prediction arm")
    require(family == expected_family and arm == expected_arm,
            "prediction family or arm differs")
    require(np.asarray(arrays["fold"]).size == 1 and np.asarray(arrays["seed"]).size == 1,
            "prediction fold or seed must be scalar")
    fold = int(np.asarray(arrays["fold"]).reshape(-1)[0])
    seed = int(np.asarray(arrays["seed"]).reshape(-1)[0])
    require(0 <= fold < 3, "prediction fold differs")
    samples = arrays["sample_key_sha256"]
    marker = arrays["marker_pos"]
    marker_cm = arrays["marker_cM"]
    probability = arrays["probabilities"]
    require(
        samples.dtype == np.dtype("|S64")
        and samples.shape == (32,)
        and np.unique(samples).size == 32
        and marker.dtype == np.dtype("<i8")
        and marker.ndim == 1
        and len(marker) > 1
        and np.all(marker[:-1] < marker[1:])
        and marker_cm.dtype == np.dtype("<f8")
        and marker_cm.shape == marker.shape
        and np.isfinite(marker_cm).all()
        and np.all(np.diff(marker_cm) >= 0)
        and probability.dtype == np.dtype("<f4")
        and probability.shape == (32, len(marker), len(STATE_NAMES))
        and np.isfinite(probability).all()
        and np.all(probability >= 0)
        and np.allclose(probability.sum(axis=2), 1.0, rtol=0, atol=5e-6),
        "prediction probability or marker axes differ",
    )
    if "state_names" in arrays:
        require(
            tuple(decode_scalar(np.asarray([value]), "state name")
                  for value in arrays["state_names"]) == STATE_NAMES,
            "prediction state order differs",
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_TRAIN_AND_PREDICT_OOF"
        and receipt.get("status") == (
            "PASS_DIAGNOSTIC_TRUTH_ENCODED_POSITIVE_CONTROL"
            if arm == "POSITIVE" else "PASS_SCORE_TRUTH_INACCESSIBLE"
        )
        and receipt.get("fold") == fold
        and receipt.get("family") == family
        and receipt.get("arm") == arm
        and receipt.get("seed") == seed
        and receipt.get("score_people") == 32
        and receipt.get("score_truth_input") is None
        and receipt.get("output_sha256") == digest,
        "prediction receipt differs",
    )
    return arrays, receipt, fold, seed, digest


def collect_oof(
    *,
    folds: Path,
    folds_receipt: Path,
    expected_folds_sha256: str,
    predictions: Sequence[Path],
    prediction_receipts: Sequence[Path],
    family: str,
    arm: str,
    output: Path,
    receipt_output: Path,
    positive_delta: float | None = None,
) -> dict[str, Any]:
    require(family in {"analytic", "tcn"}, "OOF family differs")
    require(arm in {"RE", "SHAM", "POSITIVE"},
            "RD/OFF is packed from exact F-minus-S660, not collected from fitted predictions")
    if arm == "POSITIVE":
        require(positive_delta in {0.0, 0.25, 0.5, 1.0, 2.0},
                "POSITIVE OOF needs one preregistered delta")
    else:
        require(positive_delta is None,
                "positive delta is forbidden for a production arm")
    required_seeds = TCN_SEEDS if family == "tcn" else ANALYTIC_SEEDS
    require(
        len(predictions) == len(prediction_receipts) == 3 * len(required_seeds),
        "prediction/receipt count does not cover every fold and seed",
    )
    require(not output.exists() and not receipt_output.exists(),
            "refusing to overwrite OOF outputs")
    samples, roles, _fold_document, fold_digest = _load_folds(
        folds, folds_receipt, expected_folds_sha256,
    )
    loaded: dict[tuple[int, int], tuple[dict[str, np.ndarray], str, str]] = {}
    model_contract_receipt_sha256: str | None = None
    base_contract_sha256: str | None = None
    amendment_sha256: str | None = None
    amendment_2_sha256: str | None = None
    common_marker: np.ndarray | None = None
    common_cm: np.ndarray | None = None
    common_axis_hash: np.ndarray | None = None
    positive_event_identity_sha256: str | None = None
    positive_event_masks_sha256: str | None = None
    for prediction_path, prediction_receipt in zip(
        predictions, prediction_receipts, strict=True,
    ):
        arrays, document, fold, seed, digest = _load_prediction(
            prediction_path, prediction_receipt, family, arm,
        )
        contract_receipt_sha = document.get("model_contract_receipt_sha256")
        base_sha = document.get("base_contract_sha256")
        amendment_sha = document.get("amendment_sha256")
        amendment_2_sha = document.get("amendment_2_sha256")
        if arm == "POSITIVE":
            require(
                document.get("diagnostic_only") is True
                and document.get("positive_control_delta") == positive_delta
                and "positive_delta" in arrays
                and float(np.asarray(arrays["positive_delta"]).reshape(-1)[0]) == positive_delta,
                "POSITIVE fold/delta identity differs",
            )
            event_identity = document.get("real_event_identity_sha256")
            event_masks = document.get("real_event_masks_sha256")
            require(all(isinstance(value, str) and len(value) == 64
                        for value in (event_identity, event_masks)),
                    "POSITIVE event identity provenance is absent")
            if positive_event_identity_sha256 is None:
                positive_event_identity_sha256 = event_identity
                positive_event_masks_sha256 = event_masks
            else:
                require(event_identity == positive_event_identity_sha256
                        and event_masks == positive_event_masks_sha256,
                        "POSITIVE event identity differs across fold/seed predictions")
        else:
            require(document.get("positive_control_delta") is None,
                    "production prediction carries diagnostic delta")
        require(
            all(isinstance(value, str) and len(value) == 64
                and set(value).issubset(HEX_DIGITS)
                for value in (contract_receipt_sha, base_sha, amendment_sha, amendment_2_sha)),
            "prediction lacks authenticated model-contract provenance",
        )
        if model_contract_receipt_sha256 is None:
            model_contract_receipt_sha256 = contract_receipt_sha
            base_contract_sha256 = base_sha
            amendment_sha256 = amendment_sha
            amendment_2_sha256 = amendment_2_sha
        else:
            require(
                contract_receipt_sha == model_contract_receipt_sha256
                and base_sha == base_contract_sha256
                and amendment_sha == amendment_sha256
                and amendment_2_sha == amendment_2_sha256,
                "model-contract provenance differs across fold/seed predictions",
            )
        key = (fold, seed)
        require(key not in loaded, "duplicate fold/seed prediction")
        expected_score = samples[roles[fold] == "SCORE"]
        require(np.array_equal(arrays["sample_key_sha256"], expected_score),
                "prediction sample axis differs from its SCORE fold")
        if common_marker is None:
            common_marker = arrays["marker_pos"]
            common_cm = arrays["marker_cM"]
            common_axis_hash = arrays["marker_axis_sha256"]
        else:
            require(
                np.array_equal(arrays["marker_pos"], common_marker)
                and np.array_equal(arrays["marker_cM"], common_cm)
                and np.array_equal(arrays["marker_axis_sha256"], common_axis_hash),
                "prediction marker axis differs across fold/seed files",
            )
        loaded[key] = (arrays, digest, sha256_file(prediction_receipt))
    expected_keys = {
        (fold, seed) for fold in range(3) for seed in required_seeds
    }
    require(set(loaded) == expected_keys,
            "prediction fold/seed set is incomplete or unexpected")
    require(common_marker is not None and common_cm is not None
            and common_axis_hash is not None, "no OOF prediction was loaded")

    fold_ids = np.empty(96, dtype=np.uint8)
    oof_probability = np.empty(
        (96, len(common_marker), len(STATE_NAMES)), dtype=np.float32,
    )
    assigned = np.zeros(96, dtype=np.uint8)
    for fold in range(3):
        people = np.flatnonzero(roles[fold] == "SCORE")
        stack = np.stack(
            [loaded[(fold, seed)][0]["probabilities"] for seed in required_seeds],
            axis=0,
        )
        averaged = stack.mean(axis=0, dtype=np.float64).astype(np.float32)
        require(
            np.isfinite(averaged).all()
            and np.all(averaged >= 0)
            and np.allclose(averaged.sum(axis=2), 1.0, rtol=0, atol=5e-6),
            "seed-averaged probabilities differ",
        )
        oof_probability[people] = averaged
        fold_ids[people] = fold
        assigned[people] += 1
    require(np.all(assigned == 1),
            "each person must receive exactly one OOF prediction")
    require(np.array_equal(fold_ids, np.argmax(roles == "SCORE", axis=0).astype(np.uint8)),
            "OOF fold identifiers differ")

    payload = {
        "probabilities": oof_probability,
        "sample_key_sha256": np.ascontiguousarray(samples),
        "marker_pos": np.ascontiguousarray(common_marker),
        "marker_cM": np.ascontiguousarray(common_cm),
        "marker_axis_sha256": np.ascontiguousarray(common_axis_hash),
        "fold_ids": fold_ids,
        "family": np.asarray([family]),
        "arm": np.asarray([arm]),
        "state_names": np.asarray(STATE_NAMES),
        "seed_values": np.asarray(required_seeds, dtype=np.int64),
    }
    if positive_delta is not None:
        payload["positive_delta"] = np.asarray([positive_delta], dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(output, payload)
    reopen_npz(output, payload)
    sources = []
    for fold, seed in sorted(loaded):
        _arrays, prediction_sha, receipt_sha = loaded[(fold, seed)]
        sources.append({
            "fold": fold,
            "seed": seed,
            "prediction_sha256": prediction_sha,
            "prediction_receipt_sha256": receipt_sha,
        })
    document: dict[str, Any] = {
        "schema_version": "m38b_truth_blind_oof_collection_receipt_v1",
        "stage": ("M38B_COLLECT_DIAGNOSTIC_POSITIVE_OOF"
                  if arm == "POSITIVE" else "M38B_COLLECT_TRUTH_BLIND_OOF"),
        "status": ("PASS_DIAGNOSTIC_TRUTH_ENCODED_POSITIVE_CONTROL"
                   if arm == "POSITIVE"
                   else "PASS_EXACT_ONE_OOF_PREDICTION_PER_PERSON"),
        "family": family,
        "arm": arm,
        "positive_control_delta": positive_delta,
        "diagnostic_only": arm == "POSITIVE",
        "real_event_identity_sha256": positive_event_identity_sha256,
        "real_event_masks_sha256": positive_event_masks_sha256,
        "people": 96,
        "folds": 3,
        "score_people_per_fold": 32,
        "seeds": list(required_seeds),
        "probability_aggregation": (
            "ARITHMETIC_MEAN_ACROSS_ALL_PREREGISTERED_SEEDS_NO_SELECTION"
            if len(required_seeds) > 1 else "SINGLE_PREREGISTERED_SEED"
        ),
        "person_coverage_min": int(assigned.min()),
        "person_coverage_max": int(assigned.max()),
        "truth_input": None,
        "truth_read": False,
        "truth_encoded_in_diagnostic_features": arm == "POSITIVE",
        "selector_or_checkpoint_access": False,
        "state_names": list(STATE_NAMES),
        "folds_sha256": fold_digest,
        "folds_receipt_sha256": sha256_file(folds_receipt),
        "model_contract_receipt_sha256": model_contract_receipt_sha256,
        "base_contract_sha256": base_contract_sha256,
        "amendment_sha256": amendment_sha256,
        "amendment_2_sha256": amendment_2_sha256,
        "sources": sources,
        "output_sha256": sha256_file(output),
    }
    write_exclusive_json(receipt_output, document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--folds-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-folds-sha256",
        help="Optional external pin; otherwise use the authenticated fold receipt",
    )
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--prediction-receipt", type=Path, action="append", required=True)
    parser.add_argument("--family", choices=("analytic", "tcn"), required=True)
    parser.add_argument("--arm", choices=("RE", "SHAM", "POSITIVE"), required=True)
    parser.add_argument("--positive-delta", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_folds_sha256 = args.expected_folds_sha256
    if expected_folds_sha256 is None:
        fold_receipt = json.loads(args.folds_receipt.read_text(encoding="utf-8"))
        expected_folds_sha256 = str(fold_receipt.get("output_sha256", ""))
    document = collect_oof(
        folds=args.folds,
        folds_receipt=args.folds_receipt,
        expected_folds_sha256=expected_folds_sha256,
        predictions=args.prediction,
        prediction_receipts=args.prediction_receipt,
        family=args.family,
        arm=args.arm,
        output=args.output,
        receipt_output=args.receipt,
        positive_delta=args.positive_delta,
    )
    print(json.dumps({
        "status": document["status"],
        "family": document["family"],
        "arm": document["arm"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
