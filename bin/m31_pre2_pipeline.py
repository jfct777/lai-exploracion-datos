#!/usr/bin/env python3
"""Auditable execution boundaries for the M31 PRE2 one-way experiment.

The fit and gate commands cannot accept root18 truth.  The score command is
the only entry point that can open it, after validating an OPEN receipt and
claiming the root exactly once in a durable ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m31_ordered_linear as core
import m31_pre2_contract as contract_lib
import m31_pre2_receipt as receipt_lib
import run_m31_ordered_linear as runner


WORKERS_SCREEN = (1, 4, 8)
SCIENTIFIC_ARMS = ("F0", "L", "D")
FITTED_ARMS = ("L", "D")
GLOBAL_ROOT18_LEDGER_URI = (
    "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/control/"
    "M31_ORDERED_LINEAR_DEV_PRE2.root18-20260818.CLAIMED.json"
)
AUTHORIZATION_REPORT_KEYS = {
    "schema_version", "status", "run_id", "git_commit", "container_digest",
    "contract_sha256", "execution_source_sha256", "max_cost_usd",
    "authorization_artifact_sha256", "root18_truth_accessed",
}


class Pre2Error(ValueError):
    """Raised when a PRE2 execution boundary or artifact is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Pre2Error(message)


def _gcloud_executable() -> str:
    mounted = Path("/usr/lib/google-cloud-sdk/bin/gcloud")
    return str(mounted) if mounted.is_file() else "gcloud"


def sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        runner.json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": list(values.shape), "dtype": str(values.dtype)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + values.tobytes()).hexdigest()


def fitted_payload(fitted: runner.FittedArm) -> dict[str, Any]:
    fit = {
        "alpha": fitted.alpha,
        "boundary_weight": fitted.boundary_weight,
        "cv_boundary_f1_0.2cM": fitted.cv_boundary_f1,
        "cv_false_transitions_per_cM_0.2cM": fitted.cv_false_transitions_per_cm,
        "cv_macro_ancestry_dose_mae": fitted.cv_macro_ancestry_dose_mae,
        "cv_brier": fitted.cv_brier,
        "guarded": fitted.guarded,
        "selection_status": fitted.selection_status,
        "feature_count": fitted.feature_count,
        "f0_cv_metrics": dict(fitted.f0_cv_metrics),
        "candidate_table": list(fitted.candidate_table),
        "guard_failures": list(fitted.guard_failures),
        "sufficient_stats_sha256": dict(fitted.sufficient_stats_sha256),
        "fold_stats_sha256": dict(fitted.fold_stats_sha256),
    }
    runner._validate_fit_checkpoint(fitted.arm, fit)
    model = fitted.model
    return {
        "fit": fit,
        "pre2_cv_metrics": dict(fitted.pre2_cv_metrics),
        "f0_pre2_cv_metrics": dict(fitted.f0_pre2_cv_metrics),
        "model_array_sha256": {
            "feature_mean": array_sha256(model.feature_mean),
            "feature_scale": array_sha256(model.feature_scale),
            "residual_intercept": array_sha256(model.residual_intercept),
            "coefficients": array_sha256(model.coefficients),
        },
        "selected_for_prediction": fitted.guarded,
        "no_fallback_policy": "NO_PREDICTION_WHEN_GUARDED_SET_EMPTY",
    }


def _source_hashes(paths: Sequence[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        require(resolved.is_file(), f"execution source is absent: {path}")
        hashes[resolved.name] = core.sha256_file(resolved)
    require(len(hashes) == len(paths), "execution source basenames are not unique")
    return dict(sorted(hashes.items()))


def _validate_authorization_report(
    path: Path, *, run_id: str, contract_sha256: str, git_commit: str,
    container_digest: str, execution_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(authorization, Mapping) and set(authorization) == AUTHORIZATION_REPORT_KEYS,
            "execution authorization report fields differ")
    require(authorization["schema_version"] == "1.0.0", "authorization report schema differs")
    require(authorization["status"] == "PASS_EXECUTION_AUTHORIZATION",
            "real-run authorization report is absent or invalid")
    require(authorization["run_id"] == run_id, "authorization report run ID differs")
    require(authorization["contract_sha256"] == contract_sha256,
            "authorization report contract differs")
    require(authorization["git_commit"] == git_commit, "authorization report commit differs")
    require(authorization["container_digest"] == container_digest,
            "authorization report container differs")
    require(authorization["execution_source_sha256"] == dict(execution_source_sha256),
            "authorization report execution sources differ")
    require(authorization["root18_truth_accessed"] is False,
            "authorization process accessed root18 truth")
    require(isinstance(authorization["max_cost_usd"], (int, float))
            and not isinstance(authorization["max_cost_usd"], bool)
            and float(authorization["max_cost_usd"]) > 0.0,
            "authorization report cost cap is invalid")
    require(isinstance(authorization["authorization_artifact_sha256"], str)
            and len(authorization["authorization_artifact_sha256"]) == 64
            and all(character in "0123456789abcdef"
                    for character in authorization["authorization_artifact_sha256"]),
            "authorization artifact SHA-256 is invalid")
    return dict(authorization)


def _fit_context(
    args: argparse.Namespace, workers: int, input_hashes: Mapping[str, str],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
        "stage": "ROOT17_FIT_ROOT18_TRUTH_BLIND_PREDICT",
        "workers": workers,
        "scientific_arms": list(SCIENTIFIC_ARMS),
        "fitted_arms": list(FITTED_ARMS),
        "input_sha256": dict(sorted(input_hashes.items())),
        "source_sha256": _source_hashes(args.execution_source),
        "git_commit": args.expected_git_commit,
        "container_digest": args.container_digest,
        "delta_f1": runner.PRE2_PRIMARY_DELTA_F1,
        "tau": runner.PRE2_TAU,
        "root18_truth_accepted_or_read": False,
        "execution_authorization_report_sha256": core.sha256_file(args.execution_authorization),
        "execution_authorization_artifact_sha256": authorization["authorization_artifact_sha256"],
        "authorized_cost_cap_usd": authorization["max_cost_usd"],
    }


def _fit_paths(args: argparse.Namespace) -> tuple[runner.TrainingPaths, runner.FeaturePaths]:
    return runner.TrainingPaths(**{
        key: getattr(args, f"train_root17_{key}")
        for key in ("sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit")
    }), runner.FeaturePaths(**{
        key: getattr(args, f"eval_root18_{key}")
        for key in ("sites", "target", "tree", "pools", "flare_vcf", "flare_audit")
    })


def _verify_pre2_runtime(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, str]]:
    contract = contract_lib.validate_contract(args.contract)
    hashes = {
        "contract": core.sha256_file(args.contract),
        "runner": core.sha256_file(Path(runner.__file__).resolve()),
        "core": core.sha256_file(Path(core.__file__).resolve()),
        "orchestrator": core.sha256_file(Path(__file__).resolve()),
    }
    require(hashes["contract"] == args.expected_contract_sha256, "contract SHA-256 mismatch")
    require(hashes["runner"] == args.expected_runner_sha256, "runner SHA-256 mismatch")
    require(hashes["core"] == args.expected_core_sha256, "core SHA-256 mismatch")
    require(
        len(args.expected_git_commit) == 40
        and all(character in "0123456789abcdef" for character in args.expected_git_commit),
        "expected git commit must be a lowercase 40-character SHA-1",
    )
    expected_sources = json.loads(args.expected_execution_source_sha256_json)
    observed_sources = _source_hashes([Path(__file__).resolve(), *args.execution_source])
    require(observed_sources == expected_sources,
            "staged PRE2 execution sources differ from controller-authenticated hashes")
    commit = args.expected_git_commit
    require(args.container_digest == contract["implementation"]["container_digest"],
            "container digest differs from PRE2 contract")
    return contract, commit, hashes


def fit_predict(args: argparse.Namespace) -> dict[str, Any]:
    workers, runtime = runner._validate_worker_runtime(args.workers)
    contract, commit, code_hashes = _verify_pre2_runtime(args)
    train_paths, evaluation_paths = _fit_paths(args)
    input_hashes = runner._pilot_input_hashes(args.genetic_map, train_paths, evaluation_paths)
    expected_sources = json.loads(args.expected_execution_source_sha256_json)
    authorization = _validate_authorization_report(
        args.execution_authorization,
        run_id=args.run_id,
        contract_sha256=code_hashes["contract"],
        git_commit=commit,
        container_digest=args.container_digest,
        execution_source_sha256=expected_sources,
    )
    context = _fit_context(args, workers, input_hashes, authorization)
    context["verified_code_sha256"] = code_hashes
    context["verified_git_commit"] = commit
    context["threadpool_runtime"] = runtime
    context_sha256 = sha256_payload(context)
    outdir = args.outdir
    require(not outdir.exists(), f"refusing to overwrite PRE2 worker output: {outdir}")
    outdir.mkdir(parents=True)
    runner._fsync_directory(outdir.parent)

    genetic_map = core.load_genetic_map(args.genetic_map)
    train = runner.load_feature_root(
        "root17", 20260817, train_paths.feature_paths(), genetic_map, 79791,
    )
    truth17 = runner.load_truth_bundle(train_paths, train)
    evaluation = runner.load_feature_root(
        "root18", 20260818, evaluation_paths, genetic_map, 79791,
    )
    require(train.samples == evaluation.samples, "root17/root18 sample order differs")

    f0 = runner.prepare_predictions(evaluation, None, "F0", None)
    checkpoints: dict[str, Any] = {
        "F0": runner._write_prediction_checkpoint(outdir, f0, None, context_sha256),
    }
    fit_audits: dict[str, Any] = {}
    for arm in FITTED_ARMS:
        fitted = runner.fit_arm_streaming(
            train, truth17, arm, None, contract["model"]["alphas"],
            contract["model"]["boundary_training_weights"],
            cv_seed=train.seed, workers=workers,
        )
        audit = fitted_payload(fitted)
        audit_path = outdir / f"pre2.{arm}.fit_audit.json"
        runner._atomic_json_fsync(audit_path, audit)
        fit_audits[arm] = {
            "file": audit_path.name,
            "sha256": core.sha256_file(audit_path),
            "guarded": fitted.guarded,
        }
        if fitted.guarded:
            artifact = runner.prepare_predictions(evaluation, fitted, arm, None)
            checkpoints[arm] = runner._write_prediction_checkpoint(
                outdir, artifact, fitted, context_sha256,
            )

    l_guarded = bool(fit_audits["L"]["guarded"])
    d_guarded = bool(fit_audits["D"]["guarded"])
    fingerprint = {
        "input_sha256": dict(sorted(input_hashes.items())),
        "source_sha256": context["source_sha256"],
        "delta_f1": runner.PRE2_PRIMARY_DELTA_F1,
        "tau": runner.PRE2_TAU,
        "fit_audit_semantics": {
            arm: json.loads((outdir / fit_audits[arm]["file"]).read_text(encoding="utf-8"))
            for arm in FITTED_ARMS
        },
        "prediction_semantic_sha256": {
            arm: checkpoint["prediction_semantic_sha256"]
            for arm, checkpoint in checkpoints.items()
        },
        "predicted_arms": list(checkpoints),
        "l_guarded": l_guarded,
        "d_guarded": d_guarded,
    }
    manifest = {
        "schema_version": "2.0.0",
        "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
        "status": "COMPLETE_FSYNC",
        "stage": "ROOT17_FIT_ROOT18_TRUTH_BLIND_PREDICT",
        "workers": workers,
        "context": context,
        "context_sha256": context_sha256,
        "fit_audits": fit_audits,
        "checkpoints": checkpoints,
        "scientific_fingerprint": fingerprint,
        "scientific_fingerprint_sha256": sha256_payload(fingerprint),
        "root18_truth_accepted_or_read": False,
        "decision": (
            "ELIGIBLE_FOR_ROOT17_GATE" if d_guarded else "STOP_G_D_EMPTY_NO_FALLBACK"
        ),
        "candidate_scope": (
            "PREDICTIVE_D_INCREMENT_OVER_L_NO_MECHANISM"
            if l_guarded else "RARE_COMBINED_ONLY_VS_F0"
        ),
    }
    manifest_path = outdir / "pre2.fit_predict.manifest.json"
    runner._atomic_json_fsync(manifest_path, manifest)
    return {
        "status": manifest["decision"], "manifest": str(manifest_path),
        "manifest_sha256": core.sha256_file(manifest_path),
    }


def _load_worker(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "pre2.fit_predict.manifest.json"
    require(manifest_path.is_file(), f"worker manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("status") == "COMPLETE_FSYNC"
        and manifest.get("stage") == "ROOT17_FIT_ROOT18_TRUTH_BLIND_PREDICT",
        "worker manifest status/stage differs",
    )
    require(
        isinstance(manifest.get("context"), Mapping)
        and sha256_payload(manifest["context"]) == manifest.get("context_sha256"),
        "worker context SHA-256 mismatch",
    )
    fingerprint = manifest.get("scientific_fingerprint")
    require(isinstance(fingerprint, Mapping), "worker scientific fingerprint is absent")
    require(
        sha256_payload(fingerprint) == manifest.get("scientific_fingerprint_sha256"),
        "worker scientific fingerprint SHA-256 mismatch",
    )
    require(manifest.get("root18_truth_accepted_or_read") is False, "worker accessed root18 truth")
    fit_entries = manifest.get("fit_audits", {})
    require(set(fit_entries) == set(FITTED_ARMS), "worker fit audits must be exactly L and D")
    guarded: dict[str, bool] = {}
    for arm, entry in fit_entries.items():
        path = directory / str(entry.get("file"))
        require(path.is_file() and core.sha256_file(path) == entry.get("sha256"),
                f"worker fit audit mismatch for {arm}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        runner._validate_fit_checkpoint(arm, audit.get("fit"))
        require(audit.get("selected_for_prediction") is audit["fit"]["guarded"],
                f"worker no-fallback prediction policy mismatch for {arm}")
        guarded[arm] = bool(audit["fit"]["guarded"])
    expected_checkpoints = {"F0", *({"L"} if guarded["L"] else set()),
                            *({"D"} if guarded["D"] else set())}
    require(set(manifest.get("checkpoints", {})) == expected_checkpoints,
            "worker prediction arms violate guarded-set no-fallback policy")
    for arm in expected_checkpoints:
        runner._load_prediction_checkpoint(directory, arm, str(manifest["context_sha256"]))
    return manifest


def verify_workers(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output.exists(), f"refusing to overwrite worker screen: {args.output}")
    require(len(args.worker_dir) == 3, "worker screen requires exactly three directories")
    manifests = [_load_worker(path) for path in args.worker_dir]
    by_workers = {manifest.get("workers"): manifest for manifest in manifests}
    require(set(by_workers) == set(WORKERS_SCREEN), "worker screen must contain exactly 1, 4 and 8")
    fingerprints = [by_workers[value]["scientific_fingerprint"] for value in WORKERS_SCREEN]
    require(fingerprints[0] == fingerprints[1] == fingerprints[2],
            "workers 1/4/8 differ in fit or truth-blind predictions")
    require(
        len({by_workers[value]["scientific_fingerprint_sha256"] for value in WORKERS_SCREEN}) == 1,
        "workers 1/4/8 semantic hashes differ",
    )
    chosen = by_workers[4]
    report = {
        "schema_version": "2.0.0",
        "status": "PASS_WORKERS_1_4_8_EXACT",
        "workers": list(WORKERS_SCREEN),
        "compared": [
            "sufficient_statistics", "guarded_sets", "selection", "coefficients",
            "metrics", "predictions", "semantic_hashes",
        ],
        "scientific_fingerprint_sha256": chosen["scientific_fingerprint_sha256"],
        "workers4_manifest_sha256": core.sha256_file(
            args.worker_dir[[m.get("workers") for m in manifests].index(4)]
            / "pre2.fit_predict.manifest.json"
        ),
        "root18_truth_accessed": False,
    }
    runner._atomic_json_fsync(args.output, report)
    return report


def verify_technical(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output.exists(), f"refusing to overwrite technical evidence: {args.output}")
    known = json.loads(args.known_answer.read_text(encoding="utf-8"))
    require(known.get("status") == "PASS", "known answers did not pass")
    checkpoint_sha = core.sha256_file(args.pre1_c_checkpoint)
    require(checkpoint_sha == args.expected_pre1_c_checkpoint_sha256,
            "PRE1 C checkpoint SHA-256 mismatch")
    checkpoint = json.loads(args.pre1_c_checkpoint.read_text(encoding="utf-8"))
    fit = checkpoint.get("fit")
    required_fit = {
        "alpha", "boundary_weight", "cv_boundary_f1_0.2cM", "cv_brier",
        "cv_false_transitions_per_cM_0.2cM", "cv_macro_ancestry_dose_mae",
        "feature_count", "guarded", "selection_status",
    }
    require(isinstance(fit, Mapping) and set(fit) == required_fit,
            "PRE1 C historical checkpoint fields differ")
    prediction_file_sha = core.sha256_file(args.pre1_c_prediction)
    require(prediction_file_sha == args.expected_pre1_c_prediction_sha256,
            "PRE1 C prediction file SHA-256 mismatch")
    require(checkpoint.get("prediction_file_sha256") == prediction_file_sha,
            "PRE1 C checkpoint/prediction file SHA-256 mismatch")
    stacked = np.load(args.pre1_c_prediction, mmap_mode="r", allow_pickle=False)
    require(list(stacked.shape) == checkpoint.get("shape") and str(stacked.dtype) == checkpoint.get("dtype"),
            "PRE1 C prediction shape/dtype mismatch")
    semantic = runner._prediction_sha256(
        str(checkpoint.get("root_name")), checkpoint.get("sample_ids", []),
        tuple(np.asarray(stacked[index]) for index in range(stacked.shape[0])),
    )
    require(checkpoint.get("prediction_semantic_sha256") == semantic,
            "PRE1 C semantic prediction SHA-256 mismatch")
    expected = json.loads(args.expected_pre1_c_metrics_json)
    observed = {
        "alpha": fit["alpha"], "boundary_weight": fit["boundary_weight"],
        "cv_boundary_F1_at_0.2cM": fit["cv_boundary_f1_0.2cM"],
        "cv_false_transitions_per_cM_at_0.2cM": fit["cv_false_transitions_per_cM_0.2cM"],
        "cv_macro_ancestry_dose_MAE": fit["cv_macro_ancestry_dose_mae"],
        "cv_Brier": fit["cv_brier"], "selection_status": fit["selection_status"],
    }
    require(observed == expected, "PRE1 C known-answer configuration or metrics differ")
    report = {
        "schema_version": "2.0.0", "status": "PASS_TECHNICAL_PRE2",
        "known_answers_pass": True,
        "C_exactly_reproduces_PRE1_known_answer_configuration_metrics_and_semantic_prediction_hash": True,
        "pre1_c_checkpoint_sha256": checkpoint_sha,
        "pre1_c_prediction_file_sha256": prediction_file_sha,
        "pre1_c_prediction_semantic_sha256": semantic,
        "root18_truth_accessed": False,
    }
    runner._atomic_json_fsync(args.output, report)
    return report


def verify_authorization(args: argparse.Namespace) -> dict[str, Any]:
    """Validate an external, run-specific user authorization without reading data."""
    require(not args.output.exists(), f"refusing to overwrite authorization report: {args.output}")
    payload = json.loads(args.authorization.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "experiment_id", "status", "scope", "run_id",
        "contract_sha256", "git_commit", "container_digest", "max_cost_usd",
        "execution_source_sha256", "authorized_by", "authorized_utc",
        "explicit_user_authorization",
    }
    require(isinstance(payload, Mapping) and set(payload) == expected_keys,
            "execution authorization fields differ")
    require(payload["schema_version"] == "1.0.0", "authorization schema differs")
    require(payload["experiment_id"] == "M31_ORDERED_LINEAR_DEV_PRE2",
            "authorization experiment differs")
    require(payload["status"] == "AUTHORIZED_REAL_RUN", "real run is not authorized")
    require(
        payload["scope"] == "ROOT17_FIT_GATE_AND_CONDITIONAL_SINGLE_ROOT18_SCORE",
        "authorization scope differs",
    )
    require(payload["run_id"] == args.run_id, "authorization run ID differs")
    require(payload["contract_sha256"] == core.sha256_file(args.contract),
            "authorization contract SHA-256 differs")
    require(payload["git_commit"] == args.expected_git_commit,
            "authorization git commit differs")
    require(payload["container_digest"] == args.container_digest,
            "authorization container differs")
    expected_sources = json.loads(args.expected_execution_source_sha256_json)
    require(payload["execution_source_sha256"] == expected_sources,
            "authorization execution source hashes differ")
    require(payload["authorized_by"] == "jfct777", "authorization owner differs")
    require(payload["explicit_user_authorization"] is True,
            "explicit user authorization is absent")
    require(isinstance(payload["authorized_utc"], str) and payload["authorized_utc"],
            "authorization timestamp is absent")
    require(
        isinstance(payload["max_cost_usd"], (int, float))
        and not isinstance(payload["max_cost_usd"], bool)
        and 0.0 < float(payload["max_cost_usd"]) <= args.max_cost_usd,
        "authorization cost cap is invalid or exceeds the launch cap",
    )
    report = {
        "schema_version": "1.0.0",
        "status": "PASS_EXECUTION_AUTHORIZATION",
        "run_id": args.run_id,
        "git_commit": args.expected_git_commit,
        "container_digest": args.container_digest,
        "contract_sha256": core.sha256_file(args.contract),
        "execution_source_sha256": expected_sources,
        "max_cost_usd": float(payload["max_cost_usd"]),
        "authorization_artifact_sha256": core.sha256_file(args.authorization),
        "root18_truth_accessed": False,
    }
    runner._atomic_json_fsync(args.output, report)
    return report


def known_answer(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output.exists(), f"refusing to overwrite known-answer report: {args.output}")
    contract = contract_lib.validate_contract(args.contract)
    answers = core.run_known_answer_selftest()
    synthetic = core.run_synthetic_end_to_end()
    require(answers.get("synthetic_end_to_end") == "PASS", "core known answers failed")
    require(synthetic.get("status") == "PASS", "ordered-linear synthetic test failed")
    report = {
        "schema_version": "2.0.0",
        "experiment_id": contract["experiment_id"],
        "status": "PASS",
        "stage": "KNOWN_ANSWER_NO_REAL_DATA",
        "core_known_answers": answers,
        "ordered_linear_synthetic": synthetic,
        "real_input_access": False,
        "root18_truth_accessed": False,
    }
    runner._atomic_json_fsync(args.output, report)
    return report


def _worker4_inputs(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load_worker(directory)
    require(manifest.get("workers") == 4, "gate/scorer requires the verified workers=4 artifact")
    audits = {
        arm: json.loads((directory / manifest["fit_audits"][arm]["file"]).read_text(encoding="utf-8"))
        for arm in FITTED_ARMS
    }
    metrics = {"F0": audits["D"]["f0_pre2_cv_metrics"]}
    require(metrics["F0"] == audits["L"]["f0_pre2_cv_metrics"], "L/D F0 PRE2 metrics differ")
    metrics.update({arm: audits[arm]["pre2_cv_metrics"] for arm in FITTED_ARMS})
    return manifest, audits, metrics


def _binding(args: argparse.Namespace, manifest: Mapping[str, Any]) -> dict[str, str]:
    source = manifest["context"]["source_sha256"]
    for path in (
        args.module, args.workflow, args.config, args.contract_code, args.receipt_code,
    ):
        require(source.get(path.name) == core.sha256_file(path),
                f"execution source drifted after truth-blind fit: {path.name}")
    require(
        manifest["context"]["verified_code_sha256"].get("orchestrator")
        == core.sha256_file(Path(__file__).resolve()),
        "PRE2 orchestrator drifted after truth-blind fit",
    )
    verified_code = manifest["context"]["verified_code_sha256"]
    require(verified_code.get("contract") == core.sha256_file(args.contract),
            "PRE2 contract drifted after truth-blind fit")
    require(verified_code.get("runner") == core.sha256_file(args.runner),
            "PRE2 runner drifted after truth-blind fit")
    require(verified_code.get("core") == core.sha256_file(args.core),
            "PRE2 core drifted after truth-blind fit")
    authorization = _validate_authorization_report(
        args.execution_authorization,
        run_id=args.run_id,
        contract_sha256=verified_code["contract"],
        git_commit=str(manifest["context"]["git_commit"]),
        container_digest=str(manifest["context"]["container_digest"]),
        execution_source_sha256=source,
    )
    require(
        core.sha256_file(args.execution_authorization)
        == manifest["context"]["execution_authorization_report_sha256"],
        "execution authorization report changed after truth-blind fit",
    )
    require(
        authorization["authorization_artifact_sha256"]
        == manifest["context"]["execution_authorization_artifact_sha256"],
        "execution authorization artifact changed after truth-blind fit",
    )
    return {
        "contract_sha256": core.sha256_file(args.contract),
        "git_commit": str(manifest["context"]["git_commit"]),
        "runner_sha256": core.sha256_file(args.runner),
        "core_sha256": core.sha256_file(args.core),
        "container_digest": str(manifest["context"]["container_digest"]),
        "prediction_manifest_sha256": core.sha256_file(args.worker4_dir / "pre2.fit_predict.manifest.json"),
        "context_sha256": str(manifest["context_sha256"]),
        "root17_metrics_sha256": core.sha256_file(args.root17_metrics),
        "technical_evidence_sha256": core.sha256_file(args.technical_evidence),
        "worker_screen_sha256": core.sha256_file(args.worker_screen),
        "execution_authorization_sha256": core.sha256_file(args.execution_authorization),
        "contract_code_sha256": core.sha256_file(args.contract_code),
        "receipt_code_sha256": core.sha256_file(args.receipt_code),
        "orchestrator_sha256": core.sha256_file(Path(__file__).resolve()),
        "module_sha256": source[args.module.name],
        "workflow_sha256": source[args.workflow.name],
        "config_sha256": source[args.config.name],
    }


def build_gate(args: argparse.Namespace) -> dict[str, Any]:
    require(
        not args.output.exists() and not args.open_token.exists()
        and not args.root17_metrics.exists(),
        "gate output already exists",
    )
    worker_screen = json.loads(args.worker_screen.read_text(encoding="utf-8"))
    technical = json.loads(args.technical_evidence.read_text(encoding="utf-8"))
    require(worker_screen.get("status") == "PASS_WORKERS_1_4_8_EXACT", "worker screen did not pass")
    require(technical.get("status") == "PASS_TECHNICAL_PRE2", "technical verification did not pass")
    manifest, audits, metrics = _worker4_inputs(args.worker4_dir)
    require(
        worker_screen.get("workers4_manifest_sha256")
        == core.sha256_file(args.worker4_dir / "pre2.fit_predict.manifest.json"),
        "worker screen is not bound to the selected workers=4 manifest",
    )
    runner._atomic_json_fsync(args.root17_metrics, metrics)
    technical_requirements = {
        "contract_code_input_container_hashes_match": True,
        "known_answers_pass": technical["known_answers_pass"],
        "workers_1_4_8_exact_equality_pass": True,
        "C_exactly_reproduces_PRE1_known_answer_configuration_metrics_and_semantic_prediction_hash": technical[
            "C_exactly_reproduces_PRE1_known_answer_configuration_metrics_and_semantic_prediction_hash"
        ],
        "truth_blind_prediction_manifest_fsynced": manifest["status"] == "COMPLETE_FSYNC",
    }
    claims = json.loads(args.contract.read_text(encoding="utf-8"))["claims_excluded"]
    receipt = receipt_lib.build_root17_gate_receipt(
        metrics=metrics,
        checkpoint_fits={arm: audits[arm]["fit"] for arm in FITTED_ARMS},
        technical_requirements=technical_requirements,
        binding=_binding(args, manifest),
        claims_excluded=claims,
    )
    runner._atomic_json_fsync(args.output, receipt)
    if receipt["decision"]["status"] == "OPEN_ROOT18":
        token = {
            "schema_version": "2.0.0", "status": "OPEN_ROOT18",
            "receipt_sha256": core.sha256_file(args.output),
            "receipt_semantic_sha256": receipt["receipt_semantic_sha256"],
            "run_id": args.run_id,
        }
        runner._atomic_json_fsync(args.open_token, token)
    return {"status": receipt["decision"]["status"], "receipt": str(args.output)}


def _claim_once(ledger: str | Path, run_id: str, receipt_sha256: str) -> str | Path:
    payload = json.dumps({
        "schema_version": "2.0.0", "status": "CLAIMED_ROOT18_CONSUMED",
        "run_id": run_id, "receipt_sha256": receipt_sha256,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    ledger_text = str(ledger)
    if ledger_text.startswith("gs://"):
        require(ledger_text == GLOBAL_ROOT18_LEDGER_URI, "root18 global ledger URI differs")
        with tempfile.TemporaryDirectory(prefix="m31-pre2-claim-") as temporary:
            local_claim = Path(temporary) / "claim.json"
            with local_claim.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            completed = subprocess.run(
                [_gcloud_executable(), "storage", "cp", "--if-generation-match=0",
                 str(local_claim), ledger_text],
                check=False, capture_output=True, text=True,
            )
        require(completed.returncode == 0,
                "root18 global claim already exists or could not be created atomically")
        return ledger_text
    ledger_path = Path(ledger)
    ledger_path.mkdir(parents=True, exist_ok=True)
    runner._fsync_directory(ledger_path.parent)
    claim = ledger_path / "M31_ORDERED_LINEAR_DEV_PRE2.root18-20260818.CLAIMED.json"
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise Pre2Error("root18 has already been claimed for this experiment") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    runner._fsync_directory(ledger_path)
    return claim


def _materialize_truth(source: str, expected_sha256: str, destination: Path) -> Path:
    if source.startswith("gs://"):
        subprocess.run(
            [_gcloud_executable(), "storage", "cp", source, str(destination)], check=True,
        )
        path = destination
    else:
        path = Path(source)
    require(path.is_file(), "root18 truth source is absent after the one-time claim")
    require(core.sha256_file(path) == expected_sha256, "root18 truth SHA-256 mismatch")
    return path


def _paired_bootstrap_counts(
    counts_by_arm: Mapping[str, Sequence[runner.ScoreCounts]],
) -> dict[str, Any]:
    """Bootstrap individuals once so all marginal and D-minus-control draws are paired."""
    require("D" in counts_by_arm and "F0" in counts_by_arm,
            "paired bootstrap requires D and F0")
    sample_orders = {
        arm: tuple(item.sample_id for item in counts)
        for arm, counts in counts_by_arm.items()
    }
    reference = sample_orders["D"]
    require(bool(reference) and len(set(reference)) == len(reference),
            "paired bootstrap requires unique complete diploid individuals")
    require(all(order == reference for order in sample_orders.values()),
            "paired bootstrap sample IDs/order differ across arms")

    observed = {arm: runner.summarize_counts(counts) for arm, counts in counts_by_arm.items()}
    metric_names = tuple(sorted(observed["D"]))
    require(all(tuple(sorted(metrics)) == metric_names for metrics in observed.values()),
            "paired bootstrap metric sets differ across arms")
    marginal = {
        arm: {name: [] for name in metric_names}
        for arm in counts_by_arm
    }
    comparators = tuple(arm for arm in counts_by_arm if arm != "D")
    deltas = {
        f"D_minus_{arm}": {name: [] for name in metric_names}
        for arm in comparators
    }
    rng = np.random.default_rng(core.BOOTSTRAP_SEED)
    for _replicate in range(core.BOOTSTRAP_REPLICATES):
        indexes = rng.integers(0, len(reference), size=len(reference))
        metrics = {
            arm: runner.summarize_counts([counts[int(index)] for index in indexes])
            for arm, counts in counts_by_arm.items()
        }
        for arm, arm_metrics in metrics.items():
            for name in metric_names:
                value = arm_metrics[name]
                if value is not None:
                    marginal[arm][name].append(float(value))
        for comparator in comparators:
            label = f"D_minus_{comparator}"
            for name in metric_names:
                d_value = metrics["D"][name]
                comparator_value = metrics[comparator][name]
                if d_value is not None and comparator_value is not None:
                    deltas[label][name].append(float(d_value) - float(comparator_value))

    def summarize_draws(values: Sequence[float]) -> dict[str, Any]:
        n_valid = len(values)
        return {
            "lower": float(np.quantile(values, 0.025)) if values else None,
            "upper": float(np.quantile(values, 0.975)) if values else None,
            "n_valid_replicates": n_valid,
            "n_undefined_replicates": core.BOOTSTRAP_REPLICATES - n_valid,
        }

    return {
        "unit": "complete_diploid_individual",
        "pairing": "same_sample_order_and_same_resampled_indexes_for_all_arms",
        "aggregation": "resample_individual_sufficient_counts_then_reconstruct_global_metrics",
        "replicates_requested": core.BOOTSTRAP_REPLICATES,
        "seed": core.BOOTSTRAP_SEED,
        "interval": "percentile_95_descriptive_not_a_gate",
        "sample_ids_sha256": hashlib.sha256(
            ("\n".join(reference) + "\n").encode("utf-8")
        ).hexdigest(),
        "marginal": {
            arm: {name: summarize_draws(values) for name, values in metrics.items()}
            for arm, metrics in marginal.items()
        },
        "paired_deltas": {
            label: {name: summarize_draws(values) for name, values in metrics.items()}
            for label, metrics in deltas.items()
        },
    }


def score_root18(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.outdir.exists(), f"refusing to overwrite root18 score output: {args.outdir}")
    manifest, audits, expected_metrics = _worker4_inputs(args.worker4_dir)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    technical = json.loads(args.technical_evidence.read_text(encoding="utf-8"))
    worker_screen = json.loads(args.worker_screen.read_text(encoding="utf-8"))
    require(worker_screen.get("status") == "PASS_WORKERS_1_4_8_EXACT",
            "authenticated worker screen did not pass")
    require(
        worker_screen.get("workers4_manifest_sha256")
        == core.sha256_file(args.worker4_dir / "pre2.fit_predict.manifest.json"),
        "worker screen selected a different workers=4 manifest",
    )
    technical_requirements = {
        "contract_code_input_container_hashes_match": True,
        "known_answers_pass": technical["known_answers_pass"],
        "workers_1_4_8_exact_equality_pass": True,
        "C_exactly_reproduces_PRE1_known_answer_configuration_metrics_and_semantic_prediction_hash": technical[
            "C_exactly_reproduces_PRE1_known_answer_configuration_metrics_and_semantic_prediction_hash"
        ],
        "truth_blind_prediction_manifest_fsynced": True,
    }
    claims = json.loads(args.contract.read_text(encoding="utf-8"))["claims_excluded"]
    receipt_lib.validate_root17_gate_receipt(
        receipt, expected_binding=_binding(args, manifest),
        checkpoint_fits={arm: audits[arm]["fit"] for arm in FITTED_ARMS},
        expected_metrics=expected_metrics,
        expected_technical_requirements=technical_requirements,
        expected_claims_excluded=claims,
    )
    token = json.loads(args.open_token.read_text(encoding="utf-8"))
    require(token.get("status") == "OPEN_ROOT18", "root18 opening token is not OPEN")
    require(token.get("run_id") == args.run_id, "opening token run ID differs")
    require(token.get("receipt_sha256") == core.sha256_file(args.receipt), "opening token receipt differs")
    require(args.opening_ledger == GLOBAL_ROOT18_LEDGER_URI,
            "production root18 ledger must use the immutable global GCS object")
    evaluation_paths = runner.FeaturePaths(**{
        key: getattr(args, f"eval_root18_{key}")
        for key in ("sites", "target", "tree", "pools", "flare_vcf", "flare_audit")
    })
    score_roles: dict[str, Path] = {"genetic_map": args.genetic_map}
    score_roles.update({f"root18.{key}": value for key, value in evaluation_paths.as_dict().items()})
    score_hashes = runner._authenticate_exact_subset(score_roles)
    producer_hashes = manifest["context"]["input_sha256"]
    require(all(producer_hashes.get(role) == digest for role, digest in score_hashes.items()),
            "scorer map/root18 features differ from truth-blind producer inputs")
    claim = _claim_once(args.opening_ledger, args.run_id, core.sha256_file(args.receipt))
    expected_truth = json.loads(args.contract.read_text(encoding="utf-8"))["input_sha256"]["root18.truth"]
    with tempfile.TemporaryDirectory(prefix="m31-pre2-root18-truth-") as temporary:
        truth_path = _materialize_truth(
            args.root18_truth_source, expected_truth, Path(temporary) / "root18.truth.tsv",
        )
        genetic_map = core.load_genetic_map(args.genetic_map)
        features = runner.load_feature_root("root18", 20260818, evaluation_paths, genetic_map, 79791)
        truth = runner.load_truth_bundle(runner.TrainingPaths(
            evaluation_paths.sites, evaluation_paths.target, evaluation_paths.tree,
            evaluation_paths.pools, truth_path, evaluation_paths.flare_vcf,
            evaluation_paths.flare_audit,
        ), features)
        l_guarded = bool(receipt["decision"]["l_guarded"])
        arms = ("F0", "L", "D") if l_guarded else ("F0", "D")
        metrics: dict[str, Any] = {}
        descriptive_metrics: dict[str, Any] = {}
        counts_by_arm: dict[str, Sequence[runner.ScoreCounts]] = {}
        for arm in arms:
            artifact, _checkpoint = runner._load_prediction_checkpoint(
                args.worker4_dir, arm, str(manifest["context_sha256"]),
            )
            summary, _individuals, counts = runner.score_prediction_artifact(features, truth, artifact)
            descriptive_metrics[arm] = summary
            metrics[arm] = {
                name: summary[name] for name in runner.PRE2_GATE_METRIC_NAMES
            }
            counts_by_arm[arm] = counts
        bootstrap = _paired_bootstrap_counts(counts_by_arm)
    decision = runner.evaluate_pre2_root18_decision(metrics, l_guarded=l_guarded)
    args.outdir.mkdir(parents=True)
    result = {
        "schema_version": "2.0.0", "experiment_id": "M31_ORDERED_LINEAR_DEV_PRE2",
        "stage": "ROOT18_ONE_WAY_SCORE", "decision": decision,
        "metrics": metrics,
        "descriptive_metrics": descriptive_metrics,
        "descriptive_metrics_role": "NON_GATE_NO_ANCESTRY_SPECIFIC_CLAIM",
        "bootstrap_descriptive": bootstrap,
        "claims_excluded": claims, "opening_claim": str(claim),
        "receipt_sha256": core.sha256_file(args.receipt),
        "root18_is_independent_validation": False,
        "root18_role": "single_one_way_M31_development_evaluation_not_validation_or_replication",
        "root18_used_by_prior_M29R_M30": True,
        "ASIA_is_not_NAM": True,
        "root18_consumed_no_retraining_or_reopening": True,
    }
    runner._atomic_json_fsync(args.outdir / "m31_pre2.root18.result.json", result)
    return {"status": decision["status"], "result": str(args.outdir / "m31_pre2.root18.result.json")}


def add_verified_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--container-digest", required=True)


def add_binding_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--contract-code", type=Path, required=True)
    parser.add_argument("--receipt-code", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker4-dir", type=Path, required=True)
    parser.add_argument("--root17-metrics", type=Path, required=True)
    parser.add_argument("--technical-evidence", type=Path, required=True)
    parser.add_argument("--worker-screen", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    known = sub.add_parser("known-answer")
    known.add_argument("--contract", type=Path, required=True)
    known.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("fit-predict")
    add_verified_runtime(fit)
    fit.add_argument("--workers", type=int, choices=WORKERS_SCREEN, required=True)
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--outdir", type=Path, required=True)
    fit.add_argument("--execution-source", type=Path, action="append", required=True)
    fit.add_argument("--expected-execution-source-sha256-json", required=True)
    fit.add_argument("--execution-authorization", type=Path, required=True)
    for key in ("sites", "target", "tree", "pools", "truth", "flare-vcf", "flare-audit"):
        fit.add_argument(f"--train-root17-{key}", dest=f"train_root17_{key.replace('-', '_')}", type=Path, required=True)
    for key in ("sites", "target", "tree", "pools", "flare-vcf", "flare-audit"):
        fit.add_argument(f"--eval-root18-{key}", dest=f"eval_root18_{key.replace('-', '_')}", type=Path, required=True)

    workers = sub.add_parser("verify-workers")
    workers.add_argument("--worker-dir", type=Path, action="append", required=True)
    workers.add_argument("--output", type=Path, required=True)

    technical = sub.add_parser("verify-technical")
    technical.add_argument("--known-answer", type=Path, required=True)
    technical.add_argument("--pre1-c-checkpoint", type=Path, required=True)
    technical.add_argument("--pre1-c-prediction", type=Path, required=True)
    technical.add_argument("--expected-pre1-c-checkpoint-sha256", required=True)
    technical.add_argument("--expected-pre1-c-prediction-sha256", required=True)
    technical.add_argument("--expected-pre1-c-metrics-json", required=True)
    technical.add_argument("--output", type=Path, required=True)

    authorization = sub.add_parser("verify-authorization")
    authorization.add_argument("--authorization", type=Path, required=True)
    authorization.add_argument("--contract", type=Path, required=True)
    authorization.add_argument("--run-id", required=True)
    authorization.add_argument("--expected-git-commit", required=True)
    authorization.add_argument("--container-digest", required=True)
    authorization.add_argument("--expected-execution-source-sha256-json", required=True)
    authorization.add_argument("--max-cost-usd", type=float, required=True)
    authorization.add_argument("--output", type=Path, required=True)

    gate = sub.add_parser("gate")
    add_binding_paths(gate)
    gate.add_argument("--run-id", required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--open-token", type=Path, required=True)

    score = sub.add_parser("score")
    add_binding_paths(score)
    score.add_argument("--receipt", type=Path, required=True)
    score.add_argument("--open-token", type=Path, required=True)
    score.add_argument("--opening-ledger", required=True)
    score.add_argument("--run-id", required=True)
    score.add_argument("--root18-truth-source", required=True)
    score.add_argument("--genetic-map", type=Path, required=True)
    score.add_argument("--outdir", type=Path, required=True)
    for key in ("sites", "target", "tree", "pools", "flare-vcf", "flare-audit"):
        score.add_argument(f"--eval-root18-{key}", dest=f"eval_root18_{key.replace('-', '_')}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "known-answer":
        report = known_answer(args)
    elif args.command == "fit-predict":
        report = fit_predict(args)
    elif args.command == "verify-workers":
        report = verify_workers(args)
    elif args.command == "verify-technical":
        report = verify_technical(args)
    elif args.command == "verify-authorization":
        report = verify_authorization(args)
    elif args.command == "gate":
        report = build_gate(args)
    else:
        report = score_root18(args)
    print(json.dumps(runner.json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Pre2Error, runner.RunnerError, receipt_lib.ReceiptError, core.ContractError, ValueError) as error:
        print(f"M31_PRE2_FAIL_CLOSED: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
