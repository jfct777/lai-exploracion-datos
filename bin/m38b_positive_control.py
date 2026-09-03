#!/usr/bin/env python3
"""Build one fold-specific, production-matched M38B positive control.

The control is deliberately isolated from the real RE/RD/SHAM candidates.  It
uses the existing real-event calendar and injects a known state-aligned signal
only in a diagnostic artifact.  Its magnitude is estimated from real event
likelihood contrasts among the 48 TRAIN people of the requested outer fold;
SELECT and SCORE values cannot influence that scale.

Delta zero retains the same diagnostic event geometry as positive deltas but
sets every synthetic signal to zero.  It is trained with the same TCN so the
comparison ``POS(delta) - POS(0)`` isolates the injected evidence while
holding event presence and any geometry-only behavior fixed.  The separate
production OFF/RD arm remains the exact untrained F-minus baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz, write_exclusive_json
from m37_trace_core import m34_labels_to_states


DELTA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
ROLE_COUNTS = {"TRAIN": 48, "SELECT": 16, "SCORE": 32}
EVENT_IDENTITY_MEMBERS = (
    "event_sample", "event_locus", "event_cM", "event_marker_left",
    "event_marker_right", "event_delta_left_cM", "event_delta_right_cM",
    "schedule_sample", "schedule_marker",
)
EVENT_MASK_MEMBERS = ("event_target_callable", "event_reference_callable")
AXIS_MEMBERS = (
    "sample_key_sha256", "marker_pos", "marker_cM", "marker_axis_sha256",
    "state_names",
)
HEX_DIGITS = frozenset("0123456789abcdef")


class M38BPositiveControlError(ValueError):
    """Raised when a positive-control input or invariant differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BPositiveControlError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"input is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_bundle_sha256(arrays: Mapping[str, np.ndarray], names: tuple[str, ...]) -> str:
    digest = hashlib.sha256(b"M38B_NAMED_ARRAY_BUNDLE_V1\0")
    for name in names:
        require(name in arrays, f"feature artifact lacks {name}")
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"NPZ input is not a file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _receipt_output_hash(receipt: Mapping[str, Any], artifact: Path) -> str | None:
    direct = receipt.get("output_sha256")
    if isinstance(direct, str):
        return direct
    outputs = receipt.get("outputs")
    if isinstance(outputs, dict):
        descriptor = outputs.get(artifact.name)
        if isinstance(descriptor, dict) and isinstance(descriptor.get("sha256"), str):
            return str(descriptor["sha256"])
    return None


def verify_receipt(
    artifact: Path,
    receipt_path: Path,
    allowed_stages: set[str],
    *,
    expected_arm: str | None = None,
) -> tuple[dict[str, Any], str]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("stage") in allowed_stages,
            f"{artifact.name} receipt stage differs")
    digest = sha256_file(artifact)
    require(_receipt_output_hash(receipt, artifact) == digest,
            f"{artifact.name} receipt hash differs")
    if expected_arm is not None:
        require(receipt.get("arm") == expected_arm,
                f"{artifact.name} receipt arm differs")
    return receipt, digest


def _load_truth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require({"sample_key_sha256", "marker_pos"}.issubset(archive.files),
                "positive-control truth lacks authenticated axes")
        samples = np.ascontiguousarray(archive["sample_key_sha256"])
        marker = np.ascontiguousarray(archive["marker_pos"])
        if "state_labels" in archive.files:
            labels = np.ascontiguousarray(archive["state_labels"])
        else:
            require("labels" in archive.files,
                    "positive-control truth lacks state_labels or haploid labels")
            labels = m34_labels_to_states(np.ascontiguousarray(archive["labels"]))
    require(
        labels.shape == (len(samples), len(marker))
        and np.issubdtype(labels.dtype, np.integer)
        and np.all((labels >= 0) & (labels < 6)),
        "positive-control truth dimensions or states differ",
    )
    return samples, marker, labels.astype(np.uint8, copy=False)


def _load_roles(
    path: Path, receipt_path: Path, sample_keys: np.ndarray, fold: int,
) -> tuple[np.ndarray, int, str, str]:
    receipt, digest = verify_receipt(
        path, receipt_path, {"M38B_FREEZE_OOF_ROTATION"},
    )
    require(
        receipt.get("status") == "PASS_TRUTH_BLIND_EXACT_ROTATION"
        and receipt.get("truth_read") is False
        and receipt.get("target_genotypes_read") is False,
        "positive-control fold receipt differs",
    )
    with np.load(path, allow_pickle=False) as archive:
        require(
            {"sample_key_sha256", "roles", "inner_split_seed"}.issubset(archive.files),
            "positive-control fold artifact differs",
        )
        require(np.array_equal(archive["sample_key_sha256"], sample_keys),
                "positive-control fold/sample axis differs")
        roles = np.ascontiguousarray(archive["roles"])
        seeds = np.ascontiguousarray(archive["inner_split_seed"])
    require(roles.shape == (3, 96) and seeds.shape == (3,) and 0 <= fold < 3,
            "positive-control fold dimensions differ")
    for role, count in ROLE_COUNTS.items():
        require(np.count_nonzero(roles[fold] == role) == count,
                f"positive-control {role} count differs")
    require(np.all(np.sum(roles == "SCORE", axis=0) == 1),
            "positive-control OOF coverage differs")
    train_axis = np.ascontiguousarray(sample_keys[roles[fold] == "TRAIN"])
    train_axis_hash = hashlib.sha256(train_axis.tobytes()).hexdigest()
    return roles[fold], int(seeds[fold]), digest, train_axis_hash


def _validate_feature_pair(
    real: Mapping[str, np.ndarray], rd: Mapping[str, np.ndarray],
) -> tuple[str, str, str]:
    for name in AXIS_MEMBERS:
        require(name in real and name in rd and np.array_equal(real[name], rd[name]),
                f"RE/RD {name} axis differs")
    require(
        real["sample_key_sha256"].shape == (96,)
        and np.unique(real["sample_key_sha256"]).size == 96
        and real["baseline_states"].shape[:1] == (96,)
        and real["baseline_states"].shape[2] == 6
        and real["evidence_field"].shape == real["baseline_states"].shape
        and np.array_equal(real["baseline_states"], rd["baseline_states"]),
        "positive-control feature dimensions or baselines differ",
    )
    for name in EVENT_IDENTITY_MEMBERS + EVENT_MASK_MEMBERS:
        require(name in real, f"RE feature artifact lacks {name}")
    event_count = len(real["event_sample"])
    require(event_count > 0, "positive control needs at least one real event")
    require(
        all(len(real[name]) == event_count for name in EVENT_IDENTITY_MEMBERS[:7])
        and all(len(real[name]) == event_count for name in EVENT_MASK_MEMBERS)
        and real["event_values"].shape == (event_count, 23)
        and real["event_loglik"].shape == (event_count, 6),
        "positive-control event axes differ",
    )
    require(
        len(rd["event_sample"]) == 0
        and rd["event_values"].shape == (0, 23)
        and np.count_nonzero(rd["evidence_field"]) == 0,
        "RD is not the literal event-disabled control",
    )
    axis_hash = _array_bundle_sha256(real, AXIS_MEMBERS)
    event_hash = _array_bundle_sha256(real, EVENT_IDENTITY_MEMBERS)
    mask_hash = _array_bundle_sha256(real, EVENT_MASK_MEMBERS)
    return axis_hash, event_hash, mask_hash


def robust_train_magnitude(real: Mapping[str, np.ndarray], roles: np.ndarray) -> tuple[float, int]:
    event_sample = np.asarray(real["event_sample"], dtype=np.int64)
    require(np.all((0 <= event_sample) & (event_sample < len(roles))),
            "positive-control event sample index differs")
    train_people = np.flatnonzero(roles == "TRAIN")
    train_event = np.isin(event_sample, train_people)
    likelihood = np.asarray(real["event_loglik"], dtype=np.float64)[train_event]
    require(len(likelihood) > 0 and likelihood.shape[1] == 6,
            "positive-control TRAIN has no usable event likelihoods")
    contrast = np.ptp(likelihood, axis=1)
    usable = contrast[np.isfinite(contrast) & (contrast > 0)]
    require(len(usable) > 0,
            "positive-control TRAIN robust magnitude is not identifiable")
    magnitude = float(np.median(usable))
    require(np.isfinite(magnitude) and magnitude > 0,
            "positive-control TRAIN robust magnitude differs")
    return magnitude, int(len(usable))


def _truth_aligned_values(
    real: Mapping[str, np.ndarray], truth: np.ndarray, magnitude: float, delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_sample = np.asarray(real["event_sample"], dtype=np.int64)
    left = np.asarray(real["event_marker_left"], dtype=np.int64)
    right = np.asarray(real["event_marker_right"], dtype=np.int64)
    left_distance = np.asarray(real["event_delta_left_cM"], dtype=np.float64)
    right_distance = np.asarray(real["event_delta_right_cM"], dtype=np.float64)
    require(
        np.all((0 <= left) & (left < truth.shape[1]))
        and np.all((0 <= right) & (right < truth.shape[1]))
        and np.all(np.isfinite(left_distance))
        and np.all(np.isfinite(right_distance)),
        "positive-control event/marker geometry differs",
    )
    anchor = np.where(left_distance <= right_distance, left, right)
    state = truth[event_sample, anchor].astype(np.int64, copy=False)
    amplitude = float(delta) * magnitude
    # one_hot - 1/6 is centered across states and has an exact peak-to-trough
    # contrast of one.  Thus ``amplitude`` has a direct robust-scale meaning.
    injected = np.full((len(state), 6), -amplitude / 6.0, dtype=np.float32)
    injected[np.arange(len(state)), state] = np.float32(5.0 * amplitude / 6.0)
    return injected, anchor.astype(np.int64, copy=False), state


def _inject(
    real: Mapping[str, np.ndarray], truth: np.ndarray, magnitude: float, delta: float,
) -> dict[str, np.ndarray]:
    payload = {name: np.ascontiguousarray(value).copy() for name, value in real.items()}
    injected, anchor, _state = _truth_aligned_values(real, truth, magnitude, delta)
    # Start from a geometry-only event carrier.  No real genotype, frequency,
    # uncertainty, sequence context, support, callability indicator encoded in
    # event_values, or burden is allowed into the positive-control model.  The
    # separate callability arrays are retained solely as matched masks for the
    # audit and are not consumed by the TCN.
    for name in (
        "event_genotype", "event_pooled_loglik", "event_uncertainty",
        "event_support", "event_context_7mer", "event_carrier_support",
        "event_origin_support", "event_counts",
    ):
        if name in payload:
            payload[name] = np.zeros_like(payload[name])
    for name in (
        "context_7mer_available", "carrier_support_available",
        "origin_support_available",
    ):
        if name in payload:
            payload[name] = np.zeros_like(payload[name])
    payload["event_loglik"] = injected
    values = np.zeros_like(payload["event_values"], dtype=np.float32)
    values[:, 4:10] = injected
    payload["event_values"] = values
    field = np.zeros_like(payload["evidence_field"], dtype=np.float32)
    np.add.at(
        field,
        (np.asarray(payload["event_sample"], dtype=np.int64), anchor),
        injected,
    )
    payload["evidence_field"] = field
    return payload


def build_positive_control(
    *,
    real_features: Path,
    real_receipt: Path,
    rd_features: Path,
    rd_receipt: Path,
    truth: Path,
    truth_receipt: Path,
    folds: Path,
    folds_receipt: Path,
    fold: int,
    delta: float,
    output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    require(float(delta) in DELTA_GRID and any(float(delta) == value for value in DELTA_GRID),
            "positive-control delta is outside the exact preregistered grid")
    require(not output.exists() and not receipt_output.exists(),
            "refusing to overwrite M38B positive-control output")
    _real_document, real_hash = verify_receipt(
        real_features, real_receipt,
        {"M37_TRACE_MATERIALIZE", "M38B_TRACE_MATERIALIZE"}, expected_arm="RE",
    )
    _rd_document, rd_hash = verify_receipt(
        rd_features, rd_receipt,
        {"M37_TRACE_MATERIALIZE", "M38B_TRACE_MATERIALIZE"}, expected_arm="RD",
    )
    real, rd = load_npz(real_features), load_npz(rd_features)
    axis_hash, event_hash, mask_hash = _validate_feature_pair(real, rd)
    truth_samples, truth_marker, truth_labels = _load_truth(truth)
    require(
        np.array_equal(truth_samples, real["sample_key_sha256"])
        and np.array_equal(truth_marker, real["marker_pos"]),
        "positive-control truth/features axes differ",
    )
    _truth_document, truth_hash = verify_receipt(
        truth, truth_receipt,
        {"M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660", "M38B_PARTITION_TRUTH"},
    )
    roles, inner_seed, folds_hash, train_axis_hash = _load_roles(
        folds, folds_receipt, real["sample_key_sha256"], fold,
    )
    magnitude, scale_events = robust_train_magnitude(real, roles)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _inject(real, truth_labels, magnitude, float(delta))
    require(_array_bundle_sha256(payload, AXIS_MEMBERS) == axis_hash,
            "positive-control axis changed during injection")
    require(_array_bundle_sha256(payload, EVENT_IDENTITY_MEMBERS) == event_hash,
            "positive-control event identity changed during injection")
    require(_array_bundle_sha256(payload, EVENT_MASK_MEMBERS) == mask_hash,
            "positive-control event masks changed during injection")
    if float(delta) == 0.0:
        require(
            np.count_nonzero(payload["evidence_field"]) == 0
            and np.count_nonzero(payload["event_values"]) == 0
            and np.count_nonzero(payload["event_context_7mer"]) == 0,
            "delta-zero matched positive control retains model signal",
        )
    write_deterministic_npz(output, {name: payload[name] for name in sorted(payload)})
    output_axis_hash = _array_bundle_sha256(payload, AXIS_MEMBERS)
    output_events = int(len(payload["event_sample"]))
    # Prove that constructing the diagnostic artifact did not mutate either
    # production candidate source.
    require(sha256_file(real_features) == real_hash and sha256_file(rd_features) == rd_hash,
            "positive-control construction modified a production input")
    document: dict[str, Any] = {
        "schema_version": "m38b_positive_control_receipt_v1",
        "stage": "M38B_POSITIVE_CONTROL_MATERIALIZE",
        "status": "PASS_PRODUCTION_MATCHED_DIAGNOSTIC_CONTROL",
        "arm": "POSITIVE",
        "diagnostic_only": True,
        "namespace": "DIAGNOSTIC_ONLY_NEVER_REAL_CANDIDATE",
        "fold": int(fold),
        "delta": float(delta),
        "delta_grid": list(DELTA_GRID),
        "inner_split_seed": inner_seed,
        "roles": ROLE_COUNTS,
        "scale": {
            "estimator": "MEDIAN_PER_EVENT_MAX_MINUS_MIN_LOGLIK",
            "source_role": "TRAIN_ONLY",
            "train_people": 48,
            "usable_train_events": scale_events,
            "robust_magnitude": magnitude,
            "train_sample_axis_sha256": train_axis_hash,
            "select_or_score_values_used_for_scale": False,
        },
        "truth_use": {
            "purpose": "DIAGNOSTIC_TRUTH_ALIGNED_SIGNAL_INJECTION_ONLY",
            "train_truth_used_by_model": True,
            "select_truth_used_for_checkpoint_selection": True,
            "score_truth_used_for_injection_and_later_scoring_only": True,
            "score_truth_used_to_select_real_candidate_or_checkpoint": False,
            "may_support_biological_claim": False,
        },
        "invariants": {
            "same_96_person_axis": True,
            "same_marker_axis": True,
            "same_outer_folds": True,
            "same_event_identity_as_RE_at_every_delta": True,
            "same_event_masks_as_RE_at_every_delta": True,
            "delta_zero_has_matched_events_and_zero_model_signal": bool(
                float(delta) != 0.0
                or (output_events == len(real["event_sample"])
                    and np.count_nonzero(payload["event_values"]) == 0
                    and np.count_nonzero(payload["evidence_field"]) == 0)
            ),
            "real_biological_channels_carried_into_model": [],
            "matched_nonmodel_masks": list(EVENT_MASK_MEMBERS),
            "only_delta_dependent_model_channels": [
                "event_values[4:10]", "evidence_field",
            ],
            "positive_tcn_execution_at_every_delta": (
                "SAME_ARCHITECTURE_LOSS_OPTIMIZER_SELECTION_FOLDS_AND_SEEDS"
            ),
            "positive_primary_contrast": "POS_DELTA_MINUS_POS_ZERO",
            "production_off_execution": "UNTRAINED_EXACT_F_MINUS",
            "production_inputs_modified": False,
            "required_downstream_architecture_loss_optimizer_selection": "IDENTICAL_TO_REAL_ARM",
        },
        "axis_sha256": output_axis_hash,
        "real_event_identity_sha256": event_hash,
        "real_event_masks_sha256": mask_hash,
        "events": output_events,
        "inputs": {
            "real_features_sha256": real_hash,
            "real_receipt_sha256": sha256_file(real_receipt),
            "rd_features_sha256": rd_hash,
            "rd_receipt_sha256": sha256_file(rd_receipt),
            "truth_sha256": truth_hash,
            "truth_receipt_sha256": sha256_file(truth_receipt),
            "folds_sha256": folds_hash,
            "folds_receipt_sha256": sha256_file(folds_receipt),
        },
        "output_sha256": sha256_file(output),
    }
    write_exclusive_json(receipt_output, document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-features", type=Path, required=True)
    parser.add_argument("--real-receipt", type=Path, required=True)
    parser.add_argument("--rd-features", type=Path, required=True)
    parser.add_argument("--rd-receipt", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-receipt", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--folds-receipt", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_positive_control(
        real_features=args.real_features,
        real_receipt=args.real_receipt,
        rd_features=args.rd_features,
        rd_receipt=args.rd_receipt,
        truth=args.truth,
        truth_receipt=args.truth_receipt,
        folds=args.folds,
        folds_receipt=args.folds_receipt,
        fold=args.fold,
        delta=args.delta,
        output=args.output,
        receipt_output=args.receipt,
    )
    print(json.dumps({
        "status": result["status"], "fold": result["fold"],
        "delta": result["delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
