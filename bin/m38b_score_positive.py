#!/usr/bin/env python3
"""Score the isolated M38B positive-control grid, separately from real candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m38b_score_oof import (
    _fold_vector, _text_vector, load_truth, parse_assignment, score_arm, sha256_file,
    stratified_person_bootstrap_indices,
)


DELTA_IDS = {"POS_d0": 0.0, "POS_d0p25": 0.25, "POS_d0p5": 0.5,
             "POS_d1": 1.0, "POS_d2": 2.0}


class M38BPositiveScoreError(ValueError):
    """Raised when diagnostic controls are incomplete or cross namespaces."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BPositiveScoreError(message)


def load_positive(path: Path, receipt_path: Path, logical_id: str,
                  truth_axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    people, folds, marker_cm, marker_pos = truth_axes
    delta = DELTA_IDS[logical_id]
    with np.load(path, allow_pickle=False) as archive:
        require({"probabilities", "sample_key_sha256", "fold_ids", "marker_cM",
                 "marker_pos", "state_names", "family", "seed_values", "arm",
                 "positive_delta"}.issubset(archive.files),
                "positive OOF package members differ")
        probability = np.ascontiguousarray(archive["probabilities"])
        observed_people = _text_vector(archive["sample_key_sha256"], "sample_key_sha256")
        observed_folds = _fold_vector(archive["fold_ids"], len(observed_people))
        observed_cm = np.ascontiguousarray(archive["marker_cM"])
        observed_pos = np.ascontiguousarray(archive["marker_pos"])
        state_names = tuple(
            value.decode("ascii") if isinstance(value, bytes) else str(value)
            for value in archive["state_names"].tolist()
        )
        arm = str(np.asarray(archive["arm"]).reshape(-1)[0])
        observed_delta = float(np.asarray(archive["positive_delta"]).reshape(-1)[0])
        family = str(np.asarray(archive["family"]).reshape(-1)[0])
        seeds = tuple(int(value) for value in np.asarray(archive["seed_values"]).tolist())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        arm == "POSITIVE" and family == "tcn" and seeds == (1103, 2207, 3301)
        and observed_delta == delta
        and np.array_equal(observed_people, people)
        and np.array_equal(observed_folds, folds)
        and np.array_equal(observed_cm, marker_cm)
        and np.array_equal(observed_pos, marker_pos)
        and state_names == ("AA", "AE", "AN", "EE", "EN", "NN")
        and receipt.get("stage") == "M38B_COLLECT_DIAGNOSTIC_POSITIVE_OOF"
        and receipt.get("status") == "PASS_DIAGNOSTIC_TRUTH_ENCODED_POSITIVE_CONTROL"
        and receipt.get("arm") == "POSITIVE"
        and receipt.get("family") == "tcn"
        and receipt.get("seeds") == [1103, 2207, 3301]
        and receipt.get("diagnostic_only") is True
        and receipt.get("positive_control_delta") == delta
        and receipt.get("output_sha256") == sha256_file(path),
        f"positive OOF identity differs for {logical_id}",
    )
    require(all(isinstance(receipt.get(name), str) and len(receipt[name]) == 64
                for name in ("real_event_identity_sha256", "real_event_masks_sha256")),
            f"positive OOF event identity is absent for {logical_id}")
    provenance = tuple(str(receipt.get(name, "")) for name in (
        "model_contract_receipt_sha256", "base_contract_sha256",
        "amendment_sha256", "amendment_2_sha256", "folds_sha256", "folds_receipt_sha256",
    ))
    require(all(len(value) == 64 for value in provenance),
            "positive OOF lacks contract provenance")
    return probability, provenance + (
        str(receipt["real_event_identity_sha256"]),
        str(receipt["real_event_masks_sha256"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True,
                        help="Repeat as POS_d*=path")
    parser.add_argument("--prediction-receipt", action="append", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-receipt", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=38200103)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = dict(parse_assignment(value, "=", "positive prediction")
                       for value in args.prediction)
    receipts = dict(parse_assignment(value, "=", "positive receipt")
                    for value in args.prediction_receipt)
    require(set(predictions) == set(receipts) == set(DELTA_IDS),
            "positive-control logical ID grid differs")
    truth, people, folds, marker_cm, marker_pos = load_truth(args.truth)
    fold_values = sorted(set(folds.tolist()), key=str)
    require(len(fold_values) == 3
            and all(np.count_nonzero(folds == value) == 32 for value in fold_values),
            "positive-control truth fold axis differs")
    truth_document = json.loads(args.truth_receipt.read_text(encoding="utf-8"))
    require(
        truth_document.get("stage") == "M38B_PACK_OOF_SCORE_TRUTH"
        and truth_document.get("status") == "PASS_TRUTH_SEPARATE_SCORING_BRANCH"
        and truth_document.get("output_sha256") == sha256_file(args.truth),
        "positive-control truth receipt differs",
    )
    truth_provenance = tuple(str(truth_document.get(name, "")) for name in (
        "model_contract_receipt_sha256", "base_contract_sha256",
        "amendment_sha256", "amendment_2_sha256", "folds_sha256", "folds_receipt_sha256",
    ))
    scores = {}
    provenance_rows = []
    for logical_id in DELTA_IDS:
        probability, provenance = load_positive(
            Path(predictions[logical_id]), Path(receipts[logical_id]), logical_id,
            (people, folds, marker_cm, marker_pos),
        )
        provenance_rows.append(provenance)
        scores[logical_id] = score_arm(probability, truth, marker_cm)
    core_rows = [row[:6] for row in provenance_rows]
    require(len(set(core_rows + [truth_provenance])) == 1,
            "positive-control inputs do not share contract provenance")
    require(len({row[6:] for row in provenance_rows}) == 1,
            "positive-control grid does not share real event identity and masks")
    indices = stratified_person_bootstrap_indices(
        folds, args.bootstrap_replicates, args.bootstrap_seed,
    )
    zero = scores["POS_d0"].per_person["log_loss_cm"]
    contrasts = {}
    for logical_id in tuple(DELTA_IDS)[1:]:
        delta = scores[logical_id].per_person["log_loss_cm"] - zero
        observed = float(delta.mean())
        bootstrap = delta[indices].mean(axis=1)
        upper = observed + float(np.quantile(observed - bootstrap, 0.9875))
        fold_means = {str(fold): float(delta[folds == fold].mean()) for fold in fold_values}
        passed = all(value < 0 for value in fold_means.values()) and upper < 0
        contrasts[logical_id] = {
            "delta": DELTA_IDS[logical_id],
            "contrast": f"{logical_id}-POS_d0",
            "mean_log_loss_cm_delta": observed,
            "fold_mean_deltas": fold_means,
            "favorable_folds": sum(value < 0 for value in fold_means.values()),
            "one_sided_upper_98_75": upper,
            "bonferroni_four_delta_gate": passed,
        }
    result = {
        "schema_version": "1.0.0",
        "stage": "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL",
        "status": "PASS_DIAGNOSTIC_GRID_SCORED",
        "diagnostic_only": True,
        "family": "tcn",
        "logical_ids": list(DELTA_IDS),
        "comparison": "POS_DELTA_MINUS_POS_ZERO",
        "contrasts": contrasts,
        "capacity_gate": {
            "pass": any(row["bonferroni_four_delta_gate"] for row in contrasts.values()),
            "rule": "at least one delta: favorable in 3/3 folds and one-sided 98.75% upper bound below zero",
        },
        "bootstrap": {"replicates": args.bootstrap_replicates,
                      "seed": args.bootstrap_seed, "unit": "whole person",
                      "stratified_by": "outer fold"},
        "model_contract_provenance": dict(zip((
            "model_contract_receipt_sha256", "base_contract_sha256",
            "amendment_sha256", "amendment_2_sha256", "folds_sha256", "folds_receipt_sha256",
        ), truth_provenance, strict=True)),
    }
    require(not args.output.exists(), "refusing to overwrite positive-control score")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = args.output.with_suffix(".receipt.json")
    receipt.write_text(json.dumps({
        "schema_version": "1.0.0", "stage": result["stage"],
        "status": result["status"], "diagnostic_only": True,
        "family": "tcn", "logical_ids": list(DELTA_IDS),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        **result["model_contract_provenance"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "capacity_gate": result["capacity_gate"]["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
