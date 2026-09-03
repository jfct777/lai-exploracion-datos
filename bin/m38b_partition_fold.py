#!/usr/bin/env python3
"""Create auditable M38B fold artifacts before model fitting or scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import m34_labels_to_states, require
from m38b_oof_core import slice_features


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_folds(path: Path, fold: int, sample_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as archive:
        require(
            {"sample_key_sha256", "roles", "inner_split_seed"}.issubset(archive.files),
            "M38B fold artifact differs",
        )
        require(np.array_equal(archive["sample_key_sha256"], sample_keys),
                "fold/source sample axes differ")
        roles = np.asarray(archive["roles"])
        seeds = np.asarray(archive["inner_split_seed"])
    require(roles.shape == (3, len(sample_keys)) and 0 <= fold < 3,
            "M38B fold index or dimensions differ")
    fit = np.flatnonzero(roles[fold] != "SCORE")
    score = np.flatnonzero(roles[fold] == "SCORE")
    require(
        len(fit) == 64
        and len(score) == 32
        and np.count_nonzero(roles[fold, fit] == "SELECT") == 16,
        "M38B fold role counts differ",
    )
    return fit, score, int(seeds[fold])


def authenticate_folds(path: Path, receipt_path: Path | None) -> str | None:
    if receipt_path is None:
        return None
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        document.get("stage") == "M38B_FREEZE_OOF_ROTATION"
        and document.get("status") == "PASS_TRUTH_BLIND_EXACT_ROTATION"
        and document.get("output_sha256") == sha256(path)
        and document.get("people") == 96
        and document.get("score_appearances_per_person") == 1,
        "M38B folds receipt or hash differs",
    )
    return sha256(receipt_path)


def load_all_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def partition_features(args: argparse.Namespace) -> dict[str, object]:
    folds_receipt_sha = authenticate_folds(args.folds, getattr(args, "folds_receipt", None))
    source_receipt_path = getattr(args, "source_receipt", None)
    require(source_receipt_path is not None,
            "M38B feature partition needs a source receipt")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    expected_source = getattr(args, "expected_source_sha256", None)
    expected_receipt = getattr(args, "expected_source_receipt_sha256", None)
    require(expected_source is None or sha256(args.source) == expected_source,
            "M38B pinned truth source hash differs")
    require(expected_receipt is None or sha256(source_receipt_path) == expected_receipt,
            "M38B pinned truth source receipt hash differs")
    source_stage = source_receipt.get("stage")
    source_arm = source_receipt.get("arm")
    requested_arm = getattr(args, "arm", None) or source_arm
    require(
        source_stage in {
            "M37_TRACE_MATERIALIZE", "M38B_POSITIVE_CONTROL_MATERIALIZE",
        }
        and source_arm in {"RE", "RD", "SHAM", "POSITIVE"}
        and requested_arm == source_arm
        and source_receipt.get("output_sha256") == sha256(args.source),
        "M38B materialized source receipt, arm, or hash differs",
    )
    if requested_arm == "POSITIVE":
        require(
            source_stage == "M38B_POSITIVE_CONTROL_MATERIALIZE"
            and source_receipt.get("diagnostic_only") is True
            and source_receipt.get("namespace") == "DIAGNOSTIC_ONLY_NEVER_REAL_CANDIDATE"
            and source_receipt.get("fold") == args.fold
            and source_receipt.get("delta") in {0.0, 0.25, 0.5, 1.0, 2.0},
            "M38B POSITIVE arm needs a diagnostic-only receipt",
        )
    else:
        require(source_stage == "M37_TRACE_MATERIALIZE",
                "M38B production arm needs a real materialization receipt")
    features = load_all_npz(args.source)
    require("sample_key_sha256" in features, "feature source lacks sample keys")
    fit, score, inner_seed = load_folds(
        args.folds, args.fold, features["sample_key_sha256"],
    )
    fit_payload = slice_features(features, fit)
    score_payload = slice_features(features, score)
    require(not args.fit_output.exists() and not args.score_output.exists(),
            "refusing to overwrite M38B feature partitions")
    write_deterministic_npz(args.fit_output, fit_payload)
    write_deterministic_npz(args.score_output, score_payload)
    return {
        "schema_version": "1.0.0",
        "stage": "M38B_PARTITION_FEATURES",
        "status": "PASS_TRUTH_BLIND_FEATURE_PARTITION",
        "arm": requested_arm,
        "diagnostic_only": bool(source_receipt.get("diagnostic_only", False)),
        "source_stage": source_stage,
        "source_arm": source_arm,
        "positive_control_delta": source_receipt.get("delta"),
        "real_event_identity_sha256": source_receipt.get("real_event_identity_sha256"),
        "real_event_masks_sha256": source_receipt.get("real_event_masks_sha256"),
        "fold": int(args.fold),
        "inner_split_seed": inner_seed,
        "source_sha256": sha256(args.source),
        "source_receipt_sha256": sha256(source_receipt_path),
        "folds_sha256": sha256(args.folds),
        "folds_receipt_sha256": folds_receipt_sha,
        "fit_output_sha256": sha256(args.fit_output),
        "score_output_sha256": sha256(args.score_output),
        "fit_people": 64,
        "score_people": 32,
        "truth_read": False,
    }


def _truth_payload(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require({"sample_key_sha256", "marker_pos"}.issubset(archive.files),
                "truth lacks authenticated axes")
        if "state_labels" in archive.files:
            labels = np.ascontiguousarray(archive["state_labels"])
        else:
            require("labels" in archive.files, "truth lacks ancestry labels")
            labels = m34_labels_to_states(np.ascontiguousarray(archive["labels"]))
        samples = np.ascontiguousarray(archive["sample_key_sha256"])
        marker = np.ascontiguousarray(archive["marker_pos"])
    require(labels.shape == (len(samples), len(marker)), "truth axes differ")
    return samples, marker, labels


def partition_truth(args: argparse.Namespace) -> dict[str, object]:
    folds_receipt_sha = authenticate_folds(args.folds, getattr(args, "folds_receipt", None))
    source_receipt_path = getattr(args, "source_receipt", None)
    require(source_receipt_path is not None,
            "M38B truth partition needs the alignment source receipt")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    require(
        source_receipt.get("stage") == "M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660"
        and source_receipt.get("decision") == "PASS_EXACT_SHARED_F_MINUS_S660_SCORING_GRID"
        and source_receipt.get("counts", {}).get("target_people") == 96
        and source_receipt.get("counts", {}).get("F_minus_S660") == 42326
        and source_receipt.get("outputs", {}).get(args.source.name, {}).get("sha256") == sha256(args.source),
        "M38B truth source is not authenticated by the alignment receipt",
    )
    samples, marker, labels = _truth_payload(args.source)
    fit, score, inner_seed = load_folds(args.folds, args.fold, samples)
    require(not args.fit_output.exists() and not args.score_output.exists(),
            "refusing to overwrite M38B truth partitions")
    write_deterministic_npz(args.fit_output, {
        "sample_key_sha256": samples[fit], "marker_pos": marker,
        "state_labels": labels[fit],
    })
    write_deterministic_npz(args.score_output, {
        "sample_key_sha256": samples[score], "marker_pos": marker,
        "state_labels": labels[score],
    })
    return {
        "schema_version": "1.0.0",
        "stage": "M38B_PARTITION_TRUTH",
        "status": "PASS_NON_SELECTING_TRUTH_PARTITION",
        "fold": int(args.fold),
        "inner_split_seed": inner_seed,
        "source_sha256": sha256(args.source),
        "source_receipt_sha256": sha256(source_receipt_path),
        "folds_sha256": sha256(args.folds),
        "folds_receipt_sha256": folds_receipt_sha,
        "fit_output_sha256": sha256(args.fit_output),
        "score_output_sha256": sha256(args.score_output),
        "fit_people": 64,
        "score_people": 32,
        "model_selection_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("features", "truth"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--arm", choices=("RE", "RD", "SHAM", "POSITIVE"))
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-source-receipt-sha256")
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--folds-receipt", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--score-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "features":
        require(args.source_receipt is not None and args.arm is not None,
                "feature partition needs source receipt and arm")
    else:
        require(args.source_receipt is not None and args.arm is None,
                "truth partition needs an alignment receipt and no arm")
        require(args.expected_source_sha256 is not None and
                args.expected_source_receipt_sha256 is not None,
                "truth partition needs pinned source and receipt hashes")
    require(not args.receipt.exists(), "refusing to overwrite M38B partition receipt")
    result = partition_features(args) if args.mode == "features" else partition_truth(args)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "fold": args.fold}, sort_keys=True))


if __name__ == "__main__":
    main()
