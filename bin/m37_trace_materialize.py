#!/usr/bin/env python3
"""Materialize FIT-only, phase-free TRACE inputs from authenticated NPZ factors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import (MISSING_GENOTYPE, baseline_to_states,
                            reference_state_log_likelihood, deposit_evidence,
                            diploid_state_names, require)


def load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require(required.issubset(archive.files), f"{path.name} lacks required TRACE members")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode("ascii") if isinstance(value, bytes) else str(value)
                 for value in np.asarray(values).tolist())


def verify_receipt(artifact: Path, receipt_path: Path, expected_stage: str) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("stage") == expected_stage, f"{expected_stage} receipt stage differs")
    require(receipt.get("output_sha256") == hashlib.sha256(artifact.read_bytes()).hexdigest(),
            f"{expected_stage} artifact/receipt hash differs")
    return receipt


def materialize(selected_path: Path, target_path: Path, reference_path: Path,
                f0_path: Path, marker_path: Path, marker_receipt_path: Path, arm: str,
                prior_strength: float, reference_receipt_path: Path | None = None
                ) -> dict[str, np.ndarray]:
    selected = load_npz(selected_path, {"locus_id", "cM"})
    target = load_npz(target_path, {"sample_key_sha256", "locus_id", "minor_dosage", "observed_mask"})
    reference = load_npz(reference_path, {"ancestry", "locus_id"})
    f0 = load_npz(f0_path, {"F0", "sample_key_sha256", "marker_pos"})
    marker_receipt = verify_receipt(marker_path, marker_receipt_path, "M37_BIND_MARKER_AXIS")
    marker = load_npz(marker_path, {"marker_pos", "marker_cM", "marker_axis_sha256"})
    names = decode(reference["ancestry"])
    sham_members = {"sham_seed", "sham_ancestry_permutation_sha256", "sham_source_sha256"}
    require((arm == "SHAM") == sham_members.issubset(reference),
            "SHAM arm/reference authentication differs")
    if arm == "SHAM":
        require(reference_receipt_path is not None,
                "SHAM materialization needs its generation receipt")
        verify_receipt(reference_path, reference_receipt_path, "M37_TRACE_SHAM_REFERENCE")
    else:
        require(reference_receipt_path is None,
                "a SHAM reference receipt is forbidden outside the SHAM arm")
    diploid_state_names(names)
    require(names == ("AFR", "EUR", "NAM"), "TRACE ancestry order must be AFR/EUR/NAM")
    require(np.array_equal(selected["locus_id"], target["locus_id"]) and
            np.array_equal(selected["locus_id"], reference["locus_id"]), "selected/TARGET/REF locus axes differ")
    require(np.array_equal(target["sample_key_sha256"], f0["sample_key_sha256"]), "TARGET/F0 sample axes differ")
    require(np.array_equal(marker["marker_pos"], f0["marker_pos"]),
            "bound physical marker axis differs from F0")
    require(str(marker["marker_axis_sha256"].reshape(-1)[0]) == marker_receipt.get("marker_axis_sha256"),
            "bound marker pair hash differs from receipt")
    dosage = np.asarray(target["minor_dosage"])
    observed = np.asarray(target["observed_mask"])
    require(dosage.dtype == np.dtype("|i1") and np.all(np.isin(dosage, [0, 1, 2])) and
            np.all(np.isin(observed, [0, 1])) and np.all(dosage[observed == 0] == 0),
            "TARGET dosage/missing contract differs")
    require(np.isfinite(selected["cM"]).all() and np.all(np.diff(selected["cM"]) >= 0) and
            np.isfinite(marker["marker_cM"]).all() and np.all(np.diff(marker["marker_cM"]) >= 0),
            "TRACE genetic cM axes differ")
    genotype = np.where(np.asarray(observed, dtype=bool),
                        np.asarray(target["minor_dosage"], dtype=np.uint8), MISSING_GENOTYPE)
    require(f0["F0"].ndim == 4 and f0["F0"].shape[:2] == (len(target["sample_key_sha256"]), 2) and
            f0["F0"].shape[3] == 3 and f0["F0"].shape[2] == len(marker["marker_cM"]) and
            f0["marker_pos"].shape == marker["marker_cM"].shape, "TARGET/F0/marker axes differ")
    require(genotype.shape == (len(target["sample_key_sha256"]), len(selected["locus_id"])), "TARGET dosage axes differ")
    folded = {"fold_minor_ac", "fold_callable_an"}.issubset(reference)
    aggregated = {"minor_ac", "callable_an"}.issubset(reference)
    require(folded or aggregated, "reference needs FIT fold counts or aggregate M34 minor_ac/callable_an")
    count_ac = reference["fold_minor_ac"] if folded else reference["minor_ac"]
    count_an = reference["fold_callable_an"] if folded else reference["callable_an"]
    require(count_ac.shape == count_an.shape and count_ac.shape[-2:] == (3, genotype.shape[1]),
            "FIT-only reference count axes differ")
    loglik, pooled_loglik, uncertainty, support = reference_state_log_likelihood(
        genotype, count_ac, count_an, prior_strength,
    )
    field, counts, event_index = deposit_evidence(loglik, pooled_loglik, genotype, selected["cM"], marker["marker_cM"], arm)
    baseline = baseline_to_states(f0["F0"])
    event_rows, event_loci = event_index[:, 0], event_index[:, 1]
    event_cm = np.asarray(selected["cM"], dtype=np.float64)[event_loci]
    right = np.clip(np.searchsorted(marker["marker_cM"], event_cm, side="left"), 0, len(marker["marker_cM"]) - 1)
    left = np.maximum(right - 1, 0)
    exact = np.isclose(np.asarray(marker["marker_cM"])[right], event_cm, rtol=0, atol=1e-12)
    left[exact] = right[exact]
    calendar_marker = np.unique(np.minimum(left, right)).astype(np.uint32)
    event_context = (np.asarray(selected.get("context_7mer", np.zeros(genotype.shape[1], dtype=np.uint16)))[event_loci]
                     .astype(np.uint16, copy=False))
    event_carrier = (np.asarray(selected.get("carrier_support", np.zeros(genotype.shape[1], dtype=np.float32)))[event_loci]
                     .astype(np.float32, copy=False))
    event_origin = (np.asarray(selected.get("origin_support", np.zeros(genotype.shape[1], dtype=np.float32)))[event_loci]
                    .astype(np.float32, copy=False))
    payload = {
        "sample_key_sha256": np.ascontiguousarray(target["sample_key_sha256"]),
        "marker_pos": np.ascontiguousarray(f0["marker_pos"]),
        "baseline_states": baseline,
        "evidence_field": field,
        "event_counts": counts,
        "calendar_marker": calendar_marker,
        # Scheduling metadata is kept separate from model inputs.  It remains
        # identical in RE/RD/POOLED/SHAM/GEOMETRY so every arm sees the same
        # TRAIN-only marker calendar after the deterministic person split.
        "schedule_sample": event_rows.astype(np.uint32),
        "schedule_marker": np.minimum(left, right).astype(np.uint32),
        "event_sample": event_rows.astype(np.uint32),
        "event_locus": event_loci.astype(np.uint32),
        "event_genotype": genotype[event_rows, event_loci].astype(np.uint8),
        "event_target_callable": (genotype[event_rows, event_loci] != MISSING_GENOTYPE).astype(np.uint8),
        "event_reference_callable": (support[event_rows, event_loci].max(axis=1) > 0).astype(np.uint8),
        "event_loglik": loglik[event_rows, event_loci].astype(np.float32),
        "event_pooled_loglik": pooled_loglik[event_rows, event_loci].astype(np.float32),
        "event_uncertainty": uncertainty[event_rows, event_loci].astype(np.float32),
        "event_support": support[event_rows, event_loci].astype(np.float32),
        "event_context_7mer": event_context,
        "event_carrier_support": event_carrier,
        "event_origin_support": event_origin,
        "context_7mer_available": np.asarray(["context_7mer" in selected], dtype=np.uint8),
        "carrier_support_available": np.asarray(["carrier_support" in selected], dtype=np.uint8),
        "origin_support_available": np.asarray(["origin_support" in selected], dtype=np.uint8),
        "marker_cM": np.asarray(marker["marker_cM"], dtype=np.float64),
        "marker_axis_sha256": np.asarray(marker["marker_axis_sha256"]),
        "event_cM": event_cm.astype(np.float64),
        "event_marker_left": left.astype(np.uint32),
        "event_marker_right": right.astype(np.uint32),
        "event_delta_left_cM": np.abs(event_cm - np.asarray(marker["marker_cM"])[left]).astype(np.float32),
        "event_delta_right_cM": np.abs(np.asarray(marker["marker_cM"])[right] - event_cm).astype(np.float32),
        "ancestry_names": np.asarray(names),
        "state_names": np.asarray(diploid_state_names(names)),
        "reference_frequency_policy": np.asarray(["REF_TRAIN_fold_ensemble" if folded else
                                                    "REF_TRAIN_aggregate_posterior_only_fold_ensemble_unavailable"], dtype="U80"),
        "fold_ensemble_available": np.asarray([folded], dtype=np.uint8),
        "baseline_method": np.asarray([str(np.asarray(f0.get("baseline_method", ["upstream_baseline_unspecified"])).reshape(-1)[0])]),
        "baseline_source_sha256": np.asarray([hashlib.sha256(f0_path.read_bytes()).hexdigest()]),
        "marker_axis_source_sha256": np.asarray([hashlib.sha256(marker_path.read_bytes()).hexdigest()]),
    }
    for name in ("sham_seed", "sham_ancestry_permutation_sha256", "sham_source_sha256"):
        if name in reference:
            payload[name] = np.ascontiguousarray(reference[name])
    genotype_one_hot = np.eye(4, dtype=np.float32)[payload["event_genotype"]]
    availability = np.column_stack((np.full(len(event_rows), payload["context_7mer_available"][0]),
                                    np.full(len(event_rows), payload["carrier_support_available"][0]),
                                    np.full(len(event_rows), payload["origin_support_available"][0]))).astype(np.float32)
    payload["event_values"] = np.concatenate((genotype_one_hot, payload["event_loglik"],
                                                payload["event_uncertainty"], payload["event_support"],
                                                payload["event_target_callable"][:, None],
                                                payload["event_reference_callable"][:, None],
                                                payload["event_carrier_support"][:, None],
                                                payload["event_origin_support"][:, None], availability), axis=1).astype(np.float32)
    if arm == "RD":
        # RD is an OFF control for the event TCN: do not retain event rows,
        # coordinates, genotype, support, carrier/origin or context from which
        # a biased event encoder could reconstruct signal.
        for name, value in tuple(payload.items()):
            if name.startswith("event_") and value.ndim >= 1:
                payload[name] = value[:0].copy()
        payload["event_values"] = payload["event_values"][:0].copy()
    elif arm == "GEOMETRY":
        # Preserve event coordinates for the geometry control, but no genotype,
        # likelihood, support or context content may enter the encoder.
        payload["event_values"].fill(0.0)
        payload["event_context_7mer"].fill(0)
    elif arm == "POOLED":
        payload["event_values"][:, 4:10] = payload["event_pooled_loglik"]
        # The TCN POOLED arm must not recover ancestry through three-vector
        # uncertainty/support channels retained by the ordinary RE encoder.
        payload["event_uncertainty"].fill(0.0)
        payload["event_support"].fill(0.0)
        payload["event_values"][:, 10:16] = 0.0
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference-fit-folds", type=Path, required=True)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--marker-cm", type=Path, required=True)
    parser.add_argument("--marker-axis-receipt", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--arm", choices=("RE", "RD", "POOLED", "SHAM", "GEOMETRY"), required=True)
    parser.add_argument("--beta-prior-strength", type=float, default=.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite TRACE materialization")
    payload = materialize(args.selected, args.target, args.reference_fit_folds, args.f0,
                          args.marker_cm, args.marker_axis_receipt, args.arm, args.beta_prior_strength,
                          args.reference_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(args.output, {name: payload[name] for name in sorted(payload)})
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = args.output.with_suffix(".receipt.json")
    reference_control = {}
    if "sham_ancestry_permutation_sha256" in payload:
        reference_control = {
            "type": "SHAM_PER_LOCUS_ANCESTRY_LABEL_PERMUTATION",
            "seed": int(np.asarray(payload["sham_seed"]).reshape(-1)[0]),
            "permutation_sha256": str(np.asarray(payload["sham_ancestry_permutation_sha256"]).reshape(-1)[0]),
            "source_sha256": str(np.asarray(payload["sham_source_sha256"]).reshape(-1)[0]),
        }
    receipt.write_text(json.dumps({"schema_version": "1.0.0", "stage": "M37_TRACE_MATERIALIZE",
                                   "arm": args.arm, "target_ref_disjoint": True,
                                   "target_fold_assignment": "forbidden", "output_sha256": digest,
                                   "inputs": {
                                       "selected_sha256": hashlib.sha256(args.selected.read_bytes()).hexdigest(),
                                       "target_sha256": hashlib.sha256(args.target.read_bytes()).hexdigest(),
                                       "reference_sha256": hashlib.sha256(args.reference_fit_folds.read_bytes()).hexdigest(),
                                       "F0_sha256": payload["baseline_source_sha256"][0],
                                       "marker_axis_sha256": payload["marker_axis_source_sha256"][0],
                                   },
                                   "reference_control": reference_control,
                                   "reference_receipt_sha256": (hashlib.sha256(args.reference_receipt.read_bytes()).hexdigest()
                                                                if args.reference_receipt else None),
                                   "marker_axis_receipt_sha256": hashlib.sha256(args.marker_axis_receipt.read_bytes()).hexdigest(),
                                   "physical_genetic_axis_sha256": str(payload["marker_axis_sha256"].reshape(-1)[0]),
                                   "events": int(len(payload["event_sample"]))}, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(json.dumps({"status": "PASS_TRACE_PHASE_FREE", "arm": args.arm,
                      "events": int(len(payload["event_sample"]))}, sort_keys=True))


if __name__ == "__main__":
    main()
