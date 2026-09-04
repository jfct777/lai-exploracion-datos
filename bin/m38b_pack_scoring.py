#!/usr/bin/env python3
"""Create authenticated OOF baseline and truth packages for M38B scoring.

Baseline packing never opens truth.  Truth packing is a separate command and
cannot create predictions.  Both restore the original 96-person FIT axis and
attach the single out-of-fold (OOF) fold identifier frozen before outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz, write_exclusive_json
from m37_trace_core import baseline_to_states, m34_labels_to_states, require


STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_contract_provenance(path: Path | None) -> dict[str, str]:
    require(path is not None, "M38B scoring pack needs model-contract receipt")
    document = json.loads(path.read_text(encoding="utf-8"))
    require(
        document.get("stage") == "M38B_AUTHENTICATE_MODEL_CONTRACT"
        and document.get("status") == "PASS_BASE_AND_PRE_OUTCOME_AMENDMENT_BOUND"
        and document.get("source_binding") == "DETERMINISTIC_LOAD_BEARING_SOURCE_MANIFEST"
        and isinstance(document.get("source_manifest_sha256"), str)
        and len(document["source_manifest_sha256"]) == 64
        and len(document.get("source_manifest", [])) == 28,
        "M38B scoring-pack model contract differs",
    )
    return {
        "model_contract_receipt_sha256": sha256(path),
        "base_contract_sha256": str(document["base_contract_sha256"]),
        "amendment_sha256": str(document["amendment_sha256"]),
        "amendment_2_sha256": str(document["amendment_2_sha256"]),
    }


def load_folds(path: Path, receipt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    digest = sha256(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_FREEZE_OOF_ROTATION"
        and receipt.get("status") == "PASS_TRUTH_BLIND_EXACT_ROTATION"
        and receipt.get("output_sha256") == digest
        and receipt.get("people") == 96
        and receipt.get("score_appearances_per_person") == 1,
        "M38B fold receipt differs",
    )
    with np.load(path, allow_pickle=False) as archive:
        samples = np.ascontiguousarray(archive["sample_key_sha256"])
        roles = np.ascontiguousarray(archive["roles"])
    require(samples.shape == (96,) and roles.shape == (3, 96)
            and np.all(np.sum(roles == "SCORE", axis=0) == 1),
            "M38B fold axes differ")
    fold_ids = np.argmax(roles == "SCORE", axis=0).astype(np.uint8)
    return samples, fold_ids


def load_alignment(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660"
        and receipt.get("decision") == "PASS_EXACT_SHARED_F_MINUS_S660_SCORING_GRID"
        and receipt.get("counts", {}).get("F_minus_S660") == 42326
        and receipt.get("counts", {}).get("target_people") == 96,
        "M38B alignment receipt differs",
    )
    return receipt


def load_cm(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"marker_cM"}, "common marker-cM members differ")
        cm = np.ascontiguousarray(archive["marker_cM"], dtype=np.float64)
    require(cm.shape == (42326,) and np.isfinite(cm).all()
            and np.all(np.diff(cm) >= 0), "common marker-cM axis differs")
    return cm


def _load_f0(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_key_sha256", "marker_pos", "F0"}
        require(required.issubset(archive.files), "M38B F0 members differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    require(arrays["F0"].shape == (96, 2, 42326, 3)
            and arrays["marker_pos"].shape == (42326,), "M38B F0 axes differ")
    return arrays


def pack_baselines(full_f0: Path, minus_f0: Path, marker_cm: Path,
                   alignment_receipt: Path, folds: Path, folds_receipt: Path,
                   outdir: Path, model_contract_receipt: Path | None = None) -> dict[str, Any]:
    require(not outdir.exists() or not any(outdir.iterdir()),
            "baseline output directory must be absent or empty")
    alignment = load_alignment(alignment_receipt)
    full, minus, cm = _load_f0(full_f0), _load_f0(minus_f0), load_cm(marker_cm)
    samples, fold_ids = load_folds(folds, folds_receipt)
    require(
        alignment.get("outputs", {}).get(full_f0.name, {}).get("sha256") == sha256(full_f0)
        and alignment.get("inputs", {}).get("fminus_f0") == sha256(minus_f0)
        and alignment.get("outputs", {}).get(marker_cm.name, {}).get("sha256") == sha256(marker_cm),
        "M38B baseline inputs differ from alignment receipt",
    )
    require(np.array_equal(full["sample_key_sha256"], samples)
            and np.array_equal(minus["sample_key_sha256"], samples)
            and np.array_equal(full["marker_pos"], minus["marker_pos"]),
            "M38B baseline sample or marker axes differ")
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for arm, source in (("full", full), ("minus", minus), ("RD", minus)):
        output = outdir / f"m38b_{arm}.oof.npz"
        payload = {
            "probabilities": baseline_to_states(source["F0"]),
            "sample_key_sha256": samples,
            "marker_pos": np.ascontiguousarray(source["marker_pos"], dtype=np.int64),
            "marker_cM": cm,
            "fold_ids": fold_ids,
            "arm": np.asarray([arm]),
            "state_names": np.asarray(STATE_NAMES),
        }
        write_deterministic_npz(output, payload)
        outputs[output.name] = {"sha256": sha256(output), "bytes": output.stat().st_size,
                                "source": "F_full_projected" if arm == "full" else "F_minus_S660"}
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M38B_PACK_TRUTH_BLIND_OOF_BASELINES",
        "status": "PASS_FULL_MINUS_AND_ANALYTIC_RD_PACKED",
        "truth_read": False,
        "people": 96,
        "markers": 42326,
        "folds_sha256": sha256(folds),
        "folds_receipt_sha256": sha256(folds_receipt),
        "RD_alias": "OFF_EXACT_F_MINUS_S660_NO_FIT",
        "alignment_receipt_sha256": sha256(alignment_receipt),
        "outputs": outputs,
        **model_contract_provenance(model_contract_receipt),
    }
    write_exclusive_json(outdir / "m38b_baselines.receipt.json", receipt)
    return receipt


def pack_truth(truth: Path, marker_cm: Path, alignment_receipt: Path,
               folds: Path, folds_receipt: Path, output: Path,
               model_contract_receipt: Path | None = None) -> dict[str, Any]:
    alignment = load_alignment(alignment_receipt)
    cm = load_cm(marker_cm)
    samples, fold_ids = load_folds(folds, folds_receipt)
    require(alignment.get("outputs", {}).get(truth.name, {}).get("sha256") == sha256(truth)
            and alignment.get("outputs", {}).get(marker_cm.name, {}).get("sha256") == sha256(marker_cm),
            "M38B truth inputs differ from alignment receipt")
    with np.load(truth, allow_pickle=False) as archive:
        require({"sample_key_sha256", "marker_pos", "labels"}.issubset(archive.files),
                "M38B projected truth members differ")
        truth_samples = np.ascontiguousarray(archive["sample_key_sha256"])
        marker_pos = np.ascontiguousarray(archive["marker_pos"], dtype=np.int64)
        states = m34_labels_to_states(np.ascontiguousarray(archive["labels"]))
    require(np.array_equal(truth_samples, samples) and states.shape == (96, 42326)
            and marker_pos.shape == (42326,), "M38B truth axes differ")
    payload = {
        "sample_key_sha256": samples,
        "marker_pos": marker_pos,
        "marker_cM": cm,
        "fold_ids": fold_ids,
        "state_labels": states,
    }
    write_deterministic_npz(output, payload)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M38B_PACK_OOF_SCORE_TRUTH",
        "status": "PASS_TRUTH_SEPARATE_SCORING_BRANCH",
        "model_or_checkpoint_selection_performed": False,
        "people": 96,
        "markers": 42326,
        "truth_source_sha256": sha256(truth),
        "alignment_receipt_sha256": sha256(alignment_receipt),
        "folds_sha256": sha256(folds),
        "folds_receipt_sha256": sha256(folds_receipt),
        "output_sha256": sha256(output),
        **model_contract_provenance(model_contract_receipt),
    }
    write_exclusive_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    baseline = sub.add_parser("baselines")
    baseline.add_argument("--full-f0", type=Path, required=True)
    baseline.add_argument("--minus-f0", type=Path, required=True)
    baseline.add_argument("--outdir", type=Path, required=True)
    truth = sub.add_parser("truth")
    truth.add_argument("--truth", type=Path, required=True)
    truth.add_argument("--output", type=Path, required=True)
    for command in (baseline, truth):
        command.add_argument("--marker-cm", type=Path, required=True)
        command.add_argument("--alignment-receipt", type=Path, required=True)
        command.add_argument("--folds", type=Path, required=True)
        command.add_argument("--folds-receipt", type=Path, required=True)
        command.add_argument("--model-contract-receipt", type=Path, required=True)
        command.add_argument("--expected-marker-cm-sha256", required=True)
        command.add_argument("--expected-alignment-receipt-sha256", required=True)
    baseline.add_argument("--expected-full-f0-sha256", required=True)
    baseline.add_argument("--expected-minus-f0-sha256", required=True)
    truth.add_argument("--expected-truth-sha256", required=True)
    args = parser.parse_args()
    require(sha256(args.marker_cm) == args.expected_marker_cm_sha256,
            "M38B pinned marker-cM hash differs")
    require(sha256(args.alignment_receipt) == args.expected_alignment_receipt_sha256,
            "M38B pinned alignment receipt hash differs")
    if args.mode == "baselines":
        require(sha256(args.full_f0) == args.expected_full_f0_sha256
                and sha256(args.minus_f0) == args.expected_minus_f0_sha256,
                "M38B pinned baseline hash differs")
        result = pack_baselines(args.full_f0, args.minus_f0, args.marker_cm,
                                args.alignment_receipt, args.folds,
                                args.folds_receipt, args.outdir,
                                args.model_contract_receipt)
    else:
        require(sha256(args.truth) == args.expected_truth_sha256,
                "M38B pinned truth hash differs")
        result = pack_truth(args.truth, args.marker_cm, args.alignment_receipt,
                            args.folds, args.folds_receipt, args.output,
                            args.model_contract_receipt)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
