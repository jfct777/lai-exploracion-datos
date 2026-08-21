#!/usr/bin/env python3
"""Reconcile an M31 PRE2 root17 STOP after a provenance-only gate failure.

This utility is intentionally unable to accept root18 truth or emit an OPEN
token.  It verifies the original Git blobs and the durable truth-blind
artifacts, reconstructs the frozen root17 decision, and writes a distinct
append-only erratum only when that decision is STOP_PRE2_BEFORE_ROOT18.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import m31_ordered_linear as core
import m31_pre2_receipt as receipt_lib
import run_m31_ordered_linear as runner


WORKERS = (1, 4, 8)
FITTED_ARMS = ("L", "D")
EXECUTION_SOURCE_PATHS = {
    "m31_pre2_pipeline.py": "bin/m31_pre2_pipeline.py",
    "m31_pre2_contract.py": "bin/m31_pre2_contract.py",
    "m31_pre2_receipt.py": "bin/m31_pre2_receipt.py",
    "31_ORDERED_LINEAR_PRE2.nf": "modules/31_ORDERED_LINEAR_PRE2.nf",
    "m31_ordered_linear_pre2.nf": "workflows/m31_ordered_linear_pre2.nf",
    "m31_ordered_linear_pre2.config": "conf/m31_ordered_linear_pre2.config",
}
VERIFIED_CODE_PATHS = {
    "orchestrator": "bin/m31_pre2_pipeline.py",
    "contract": "conf/m31_ordered_linear_pre2_preregistration.json",
    "runner": "bin/run_m31_ordered_linear.py",
    "core": "bin/m31_ordered_linear.py",
}
SCREEN_KEYS = {
    "schema_version", "status", "workers", "compared",
    "scientific_fingerprint_sha256", "workers4_manifest_sha256",
    "root18_truth_accessed",
}


class ReconciliationError(ValueError):
    """Raised when an M31 STOP cannot be reconstructed exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        runner.json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(runner.json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required artifact is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, Mapping), f"JSON artifact is not an object: {path}")
    return dict(payload)


def _original_blob_sha256(
    repository: Path | None, source_snapshot: Path | None,
    commit: str, relative_path: str,
) -> str:
    if source_snapshot is not None:
        path = source_snapshot / relative_path
        require(path.is_file(), f"original source snapshot blob is absent: {relative_path}")
        return core.sha256_file(path)
    require(repository is not None, "Git repository or source snapshot is required")
    completed = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
        check=False, capture_output=True,
    )
    require(completed.returncode == 0,
            f"original Git blob is unavailable: {commit}:{relative_path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def authorized_sources_from_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    context = manifest.get("context")
    require(isinstance(context, Mapping), "worker context is absent")
    staged = context.get("source_sha256")
    verified = context.get("verified_code_sha256")
    require(isinstance(staged, Mapping), "worker source hashes are absent")
    require(isinstance(verified, Mapping), "worker verified code hashes are absent")
    require(set(staged) == set(EXECUTION_SOURCE_PATHS) - {"m31_pre2_pipeline.py"},
            "worker staged source set differs from the frozen five-source set")
    require("m31_pre2_pipeline.py" not in staged,
            "worker staged sources duplicate the orchestrator")
    orchestrator = verified.get("orchestrator")
    require(isinstance(orchestrator, str) and len(orchestrator) == 64,
            "worker orchestrator SHA-256 is invalid")
    sources = {str(name): str(value) for name, value in staged.items()}
    sources["m31_pre2_pipeline.py"] = orchestrator
    require(set(sources) == set(EXECUTION_SOURCE_PATHS),
            "reconstructed authorization source set differs")
    return dict(sorted(sources.items()))


def _load_stop_worker(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "pre2.fit_predict.manifest.json"
    manifest = _load_json(manifest_path)
    require(manifest.get("status") == "COMPLETE_FSYNC", "worker is not complete")
    require(manifest.get("stage") == "ROOT17_FIT_ROOT18_TRUTH_BLIND_PREDICT",
            "worker stage differs")
    require(manifest.get("root18_truth_accepted_or_read") is False,
            "worker accessed root18 truth")
    context = manifest.get("context")
    fingerprint = manifest.get("scientific_fingerprint")
    require(isinstance(context, Mapping) and sha256_payload(context) == manifest.get("context_sha256"),
            "worker context SHA-256 mismatch")
    require(isinstance(fingerprint, Mapping)
            and sha256_payload(fingerprint) == manifest.get("scientific_fingerprint_sha256"),
            "worker scientific fingerprint SHA-256 mismatch")
    require(manifest.get("decision") == "STOP_G_D_EMPTY_NO_FALLBACK",
            "worker did not record the frozen empty-guarded-set STOP")
    entries = manifest.get("fit_audits")
    require(isinstance(entries, Mapping) and set(entries) == set(FITTED_ARMS),
            "worker fit audit set differs")
    audits: dict[str, Any] = {}
    for arm in FITTED_ARMS:
        entry = entries[arm]
        audit_path = directory / str(entry.get("file"))
        require(audit_path.is_file() and core.sha256_file(audit_path) == entry.get("sha256"),
                f"worker fit audit mismatch for {arm}")
        audit = _load_json(audit_path)
        fit = audit.get("fit")
        runner._validate_fit_checkpoint(arm, fit)
        require(fit["guarded"] is False and audit.get("selected_for_prediction") is False,
                f"{arm} is not an unguarded no-fallback fit")
        require(len(fit["candidate_table"]) == 18
                and sum(bool(row["guarded"]) for row in fit["candidate_table"]) == 0,
                f"{arm} does not contain exactly 18 rejected configurations")
        audits[arm] = audit
    require(set(manifest.get("checkpoints", {})) == {"F0"},
            "STOP worker contains a forbidden L or D prediction")
    return manifest, audits


def verify_worker_screen(
    screen: Mapping[str, Any], manifests: Mapping[int, Mapping[str, Any]],
    worker4_manifest_path: Path,
) -> None:
    require(set(screen) == SCREEN_KEYS, "worker screen fields differ")
    require(screen.get("schema_version") == "2.0.0", "worker screen schema differs")
    require(screen.get("status") == "PASS_WORKERS_1_4_8_EXACT",
            "worker screen did not pass")
    require(screen.get("workers") == list(WORKERS), "worker screen set differs")
    require(screen.get("root18_truth_accessed") is False,
            "worker screen reports root18 access")
    fingerprints = [manifests[value]["scientific_fingerprint"] for value in WORKERS]
    require(fingerprints[0] == fingerprints[1] == fingerprints[2],
            "workers 1/4/8 scientific fingerprints differ")
    fingerprint_sha = manifests[4]["scientific_fingerprint_sha256"]
    require(screen.get("scientific_fingerprint_sha256") == fingerprint_sha,
            "worker screen fingerprint is not bound to workers=4")
    require(screen.get("workers4_manifest_sha256") == core.sha256_file(worker4_manifest_path),
            "worker screen is not bound to the workers=4 manifest")


def require_stop_only(receipt: Mapping[str, Any]) -> None:
    decision = receipt.get("decision")
    require(isinstance(decision, Mapping), "reconstructed gate decision is absent")
    require(decision.get("status") == "STOP_PRE2_BEFORE_ROOT18",
            "STOP-only reconciliation refuses a decision that could open root18")


def _verify_original_sources(
    repository: Path | None, source_snapshot: Path | None,
    commit: str, manifest: Mapping[str, Any],
    authorization: Mapping[str, Any], contract: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    require(len(commit) == 40 and all(character in "0123456789abcdef" for character in commit),
            "original commit is not a lowercase 40-character SHA-1")
    require(manifest["context"].get("git_commit") == commit,
            "worker commit differs from the requested original commit")
    require(authorization.get("git_commit") == commit,
            "authorization commit differs from the requested original commit")
    sources = authorized_sources_from_manifest(manifest)
    require(authorization.get("execution_source_sha256") == sources,
            "authorization report execution sources differ after exact reconstruction")
    observed_sources = {
        name: _original_blob_sha256(repository, source_snapshot, commit, path)
        for name, path in EXECUTION_SOURCE_PATHS.items()
    }
    require(observed_sources == sources, "original Git execution source blobs differ")
    verified = manifest["context"]["verified_code_sha256"]
    observed_verified = {
        name: _original_blob_sha256(repository, source_snapshot, commit, path)
        for name, path in VERIFIED_CODE_PATHS.items()
    }
    require(observed_verified == verified, "original Git verified-code blobs differ")
    require(core.sha256_file(contract) == observed_verified["contract"],
            "staged contract differs from the original Git blob")
    return observed_sources, observed_verified


def reconcile(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), f"refusing to overwrite reconciliation: {args.output_dir}")
    require(len(args.worker_dir) == 3, "reconciliation requires workers 1, 4 and 8")
    authorization = _load_json(args.authorization_report)
    screen = _load_json(args.worker_screen)
    technical = _load_json(args.technical_evidence)
    contract = _load_json(args.contract)

    loaded = [_load_stop_worker(path) for path in args.worker_dir]
    by_workers = {int(manifest.get("workers")): (manifest, audits, path)
                  for (manifest, audits), path in zip(loaded, args.worker_dir)}
    require(set(by_workers) == set(WORKERS), "worker directories are not exactly 1, 4 and 8")
    manifests = {workers: by_workers[workers][0] for workers in WORKERS}
    worker4_manifest_path = by_workers[4][2] / "pre2.fit_predict.manifest.json"
    verify_worker_screen(screen, manifests, worker4_manifest_path)
    manifest, audits, _worker4_dir = by_workers[4]

    require(authorization.get("schema_version") == "1.0.0"
            and authorization.get("status") == "PASS_EXECUTION_AUTHORIZATION",
            "execution authorization report is invalid")
    require(authorization.get("run_id") == args.run_id, "authorization run ID differs")
    require(authorization.get("root18_truth_accessed") is False,
            "authorization process accessed root18 truth")
    require(core.sha256_file(args.authorization_report)
            == manifest["context"].get("execution_authorization_report_sha256"),
            "authorization report changed after the truth-blind fit")
    require(authorization.get("authorization_artifact_sha256")
            == manifest["context"].get("execution_authorization_artifact_sha256"),
            "authorization artifact binding differs")
    require(float(authorization.get("max_cost_usd"))
            == float(manifest["context"].get("authorized_cost_cap_usd")),
            "authorized cost cap differs from the worker manifest")
    observed_sources, observed_verified = _verify_original_sources(
        args.repository, args.source_snapshot_dir, args.original_commit,
        manifest, authorization, args.contract,
    )
    require(authorization.get("contract_sha256") == observed_verified["contract"],
            "authorization contract differs")
    require(authorization.get("container_digest") == manifest["context"].get("container_digest"),
            "authorization container differs")

    require(technical.get("status") == "PASS_TECHNICAL_PRE2"
            and technical.get("known_answers_pass") is True
            and technical.get(
                "C_exactly_reproduces_PRE1_known_answer_configuration_metrics_and_semantic_prediction_hash"
            ) is True
            and technical.get("root18_truth_accessed") is False,
            "technical evidence did not pass or accessed root18 truth")
    f0 = audits["D"]["f0_pre2_cv_metrics"]
    require(f0 == audits["L"]["f0_pre2_cv_metrics"], "L/D F0 metrics differ")
    metrics = {"F0": f0, **{arm: audits[arm]["pre2_cv_metrics"] for arm in FITTED_ARMS}}
    metrics = {
        arm: {name: values[name] for name in runner.PRE2_GATE_METRIC_NAMES}
        for arm, values in metrics.items()
    }
    metrics_sha = hashlib.sha256(json_bytes(metrics)).hexdigest()
    technical_requirements = {
        "contract_code_input_container_hashes_match": True,
        "known_answers_pass": True,
        "workers_1_4_8_exact_equality_pass": True,
        "C_exactly_reproduces_PRE1_known_answer_configuration_metrics_and_semantic_prediction_hash": True,
        "truth_blind_prediction_manifest_fsynced": True,
    }
    binding = {
        "contract_sha256": observed_verified["contract"],
        "git_commit": args.original_commit,
        "runner_sha256": observed_verified["runner"],
        "core_sha256": observed_verified["core"],
        "container_digest": str(manifest["context"]["container_digest"]),
        "prediction_manifest_sha256": core.sha256_file(worker4_manifest_path),
        "context_sha256": str(manifest["context_sha256"]),
        "root17_metrics_sha256": metrics_sha,
        "technical_evidence_sha256": core.sha256_file(args.technical_evidence),
        "worker_screen_sha256": core.sha256_file(args.worker_screen),
        "execution_authorization_sha256": core.sha256_file(args.authorization_report),
        "contract_code_sha256": observed_sources["m31_pre2_contract.py"],
        "receipt_code_sha256": observed_sources["m31_pre2_receipt.py"],
        "orchestrator_sha256": observed_verified["orchestrator"],
        "module_sha256": observed_sources["31_ORDERED_LINEAR_PRE2.nf"],
        "workflow_sha256": observed_sources["m31_ordered_linear_pre2.nf"],
        "config_sha256": observed_sources["m31_ordered_linear_pre2.config"],
    }
    reconstructed = receipt_lib.build_root17_gate_receipt(
        metrics=metrics,
        checkpoint_fits={arm: audits[arm]["fit"] for arm in FITTED_ARMS},
        technical_requirements=technical_requirements,
        binding=binding,
        claims_excluded=contract["claims_excluded"],
    )
    require_stop_only(reconstructed)
    body = {
        "schema_version": "1.0.0",
        "status": "RECONCILED_STOP_PRE2_BEFORE_ROOT18",
        "artifact_role": "POST_RUN_ERRATUM_NOT_ORIGINAL_GATE_RECEIPT",
        "run_id": args.run_id,
        "original_git_commit": args.original_commit,
        "original_gate_not_completed_due_provenance_binding_bug": True,
        "bug": "authorization_six_sources_compared_to_manifest_five_without_separate_orchestrator",
        "reconstructed_decision": reconstructed,
        "worker_screen_status": screen["status"],
        "workers": list(WORKERS),
        "reconciler_sha256": core.sha256_file(Path(__file__).resolve()),
        "input_sha256": {
            "authorization_report": core.sha256_file(args.authorization_report),
            "worker_screen": core.sha256_file(args.worker_screen),
            "technical_evidence": core.sha256_file(args.technical_evidence),
            "worker1_manifest": core.sha256_file(by_workers[1][2] / "pre2.fit_predict.manifest.json"),
            "worker4_manifest": core.sha256_file(worker4_manifest_path),
            "worker8_manifest": core.sha256_file(by_workers[8][2] / "pre2.fit_predict.manifest.json"),
        },
        "root18_truth_accepted_read_or_claimed_by_reconciler": False,
        "open_token_emitted": False,
    }
    report = {**body, "reconciliation_semantic_sha256": sha256_payload(body)}

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=str(args.output_dir.parent),
    ))
    try:
        runner._atomic_json_fsync(temporary / "m31_pre2.root17.metrics.reconciled.json", metrics)
        runner._atomic_json_fsync(
            temporary / "m31_pre2.root17.stop_receipt.reconstructed.json", reconstructed,
        )
        runner._atomic_json_fsync(temporary / "m31_pre2.STOP.erratum.json", report)
        os.replace(temporary, args.output_dir)
        runner._fsync_directory(args.output_dir.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-report", type=Path, required=True)
    parser.add_argument("--worker-screen", type=Path, required=True)
    parser.add_argument("--technical-evidence", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repository", type=Path)
    source.add_argument("--source-snapshot-dir", type=Path)
    parser.add_argument("--original-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = reconcile(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconciliationError, runner.RunnerError, receipt_lib.ReceiptError,
            core.ContractError, ValueError, KeyError, TypeError) as error:
        print(f"M31_PRE2_STOP_RECONCILIATION_FAIL_CLOSED: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
