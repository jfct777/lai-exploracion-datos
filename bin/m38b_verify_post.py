#!/usr/bin/env python3
"""Fail-closed, post-hoc verification of a completed M38B run.

This program never trains a model.  It reopens the published out-of-fold (OOF)
probabilities and sealed truth, recomputes every reported score, bootstrap draw,
gate, and final decision, and binds the complete published run to a new receipt.
It is intentionally separate from the workflow that produces M38B outcomes so
that a running analysis is never modified in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m34_parse_flare_truth import sha256_file
from m38b_positive_control import (
    AXIS_MEMBERS,
    EVENT_IDENTITY_MEMBERS,
    EVENT_MASK_MEMBERS,
    _array_bundle_sha256,
    load_npz,
)
from m38b_score_oof import (
    ANCESTRY_NAMES,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    PER_PERSON_METRICS,
    STATE_NAMES,
    analyse_files,
    load_prediction,
    load_truth,
    score_arm,
    stratified_person_bootstrap_indices,
    verify_scoring_receipts,
)
from m38b_score_positive import DELTA_IDS, load_positive
from m38b_validate_model_contract import LOAD_BEARING_SOURCE_NAMES


EXPECTED_PEOPLE = 96
EXPECTED_MARKERS = 42_326
EXPECTED_FOLDS = ("0", "1", "2")
EXPECTED_FOLD_SIZE = 32
EXPECTED_BOOTSTRAP_REPLICATES = 10_000
EXPECTED_BOOTSTRAP_SEED = 38_200_103
EXPECTED_PUBLISHED_FILES = 356
EXPECTED_GIT_COMMIT = "8eb46d54be81d89c6adfd43979c13d44f49b7b78"
EXPECTED_CONTRASTS = (
    ("full-minus", "full", "minus"),
    ("RE-RD", "RE", "RD"),
    ("RE-SHAM", "RE", "SHAM"),
    ("RE-full", "RE", "full"),
)
EXPECTED_ARMS = ("full", "minus", "RD", "RE", "SHAM")
EXPECTED_POSITIVE_IDS = ("POS_d0", "POS_d0p25", "POS_d0p5", "POS_d1", "POS_d2")
POSITIVE_ZERO_CHANNELS = (
    "event_genotype",
    "event_pooled_loglik",
    "event_uncertainty",
    "event_support",
    "event_context_7mer",
    "event_carrier_support",
    "event_origin_support",
    "event_counts",
    "context_7mer_available",
    "carrier_support_available",
    "origin_support_available",
)
EXPECTED_BASE_CONTRACT_SHA256 = "b8673af7c76b5844c340e6c21286abc8a26be37195e11e304da621534b964c00"
EXPECTED_AMENDMENT_SHA256 = "26bfec992f977cfd15033376a211dd9436a71e0b1542186a1ebf3bf335622c15"
EXPECTED_AMENDMENT_2_SHA256 = "472607c4f95d782ae4a540decb994adcee09a2469ac7aae7b3df27a66b61bda8"
EXPECTED_MODEL_CONTRACT_RECEIPT_SHA256 = (
    "c60ea41a9ca39b2dc7e360b4d7a48f98c8b222325885a1dfa39408a1842e3337"
)
EXPECTED_RUN_PROVENANCE_SHA256 = (
    "fd598e1837de09a41b541078ed08bc82bde16484c4c229c9a22b6356e4409101"
)
PROVENANCE_FIELDS = (
    "model_contract_receipt_sha256",
    "base_contract_sha256",
    "amendment_sha256",
    "amendment_2_sha256",
    "folds_sha256",
    "folds_receipt_sha256",
)
ERROR_VETO_METRICS = (
    "brier_cm",
    "ancestry_proportion_mae_macro_cm",
    "ancestry_proportion_mae_nam_cm",
    "ancestry_proportion_mae_nam_truth_present_cm",
    "false_transitions_per_morgan_0_2cm",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class M38BPostVerificationError(ValueError):
    """Raised whenever a published M38B invariant cannot be proved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BPostVerificationError(message)


def require_exact_bool(value: Any, name: str) -> bool:
    require(type(value) is bool, f"{name} must be a JSON boolean")
    return value


def require_exact_int(value: Any, expected: int, name: str) -> int:
    require(type(value) is int and value == expected, f"{name} must equal {expected}")
    return value


def require_finite_number(value: Any, name: str) -> float:
    require(type(value) in (int, float) and not isinstance(value, bool),
            f"{name} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite")
    return result


def require_hash(value: Any, name: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            f"{name} must be a lowercase SHA-256")
    return value


def _reject_json_constant(value: str) -> None:
    raise M38BPostVerificationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON input is not a file: {path}")
    document = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    require(isinstance(document, dict), f"JSON root must be an object: {path}")
    return document


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")


@dataclass
class HashLedger:
    """Byte-level manifest of every artifact used by the verifier."""

    run_dir: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def audit(self, path: Path) -> str:
        resolved_root = self.run_dir.resolve()
        resolved = path.resolve()
        require(path.is_file(), f"audit input is not a file: {path}")
        require(resolved.is_relative_to(resolved_root),
                f"audit input escapes run directory: {path}")
        relative = resolved.relative_to(resolved_root).as_posix()
        digest = sha256_file(path)
        row = {"path": relative, "bytes": path.stat().st_size, "sha256": digest}
        if relative in self.entries:
            require(self.entries[relative] == row, f"artifact changed while auditing: {relative}")
        self.entries[relative] = row
        return digest

    def document(self) -> list[dict[str, Any]]:
        return [self.entries[name] for name in sorted(self.entries)]


def expected_relative_paths() -> set[str]:
    """Return the exact 356-file publication contract for M38B run c."""
    paths = {
        "prelaunch/m38b.model_contract.receipt.json",
        "prelaunch/run_provenance.receipt.json",
        "contract/m38b.model_contract.receipt.json",
        "axes/m38b.marker_axis.npz",
        "axes/m38b.marker_axis.receipt.json",
        "controls/sham/m38b.strict_sham.reference.npz",
        "controls/sham/m38b.strict_sham.reference.receipt.json",
        "folds/m38b.folds.npz",
        "folds/m38b.folds.receipt.json",
        "controls/positive/score/m38b.positive.metrics.json",
        "controls/positive/score/m38b.positive.metrics.receipt.json",
        "decision/m38b.final_decision.json",
        "decision/m38b.final_decision.receipt.json",
        "score/m38b.score.truth.npz",
        "score/m38b.score.truth.receipt.json",
    }
    factor_root = "factors/m38b_primary_factors"
    paths.update({
        f"{factor_root}/m38b_primary_selected_loci.npz",
        f"{factor_root}/m38b_primary_target_rare_diploid.npz",
        f"{factor_root}/m38b_primary_reference_rare_summary.npz",
        f"{factor_root}/m38b_primary_factor_subset.receipt.json",
    })
    for arm in ("RE", "RD", "SHAM"):
        paths.add(f"features/m38b.{arm}.trace.npz")
        paths.add(f"features/m38b.{arm}.trace.receipt.json")
    for family in ("analytic", "tcn"):
        paths.update({
            f"score/m38b.{family}.metrics.json",
            f"score/m38b.{family}.metrics.per_person.npz",
            f"score/m38b.{family}.metrics.receipt.json",
        })
    baseline_root = "predictions/baselines/m38b_packed_baselines"
    paths.update({
        f"{baseline_root}/m38b_full.oof.npz",
        f"{baseline_root}/m38b_minus.oof.npz",
        f"{baseline_root}/m38b_RD.oof.npz",
        f"{baseline_root}/m38b_baselines.receipt.json",
    })
    identities = ("RE", "SHAM") + EXPECTED_POSITIVE_IDS
    for identity in identities:
        if identity in EXPECTED_POSITIVE_IDS:
            for fold in range(3):
                stem = f"controls/positive/features/m38b.{identity}.fold{fold}"
                paths.update({f"{stem}.npz", f"{stem}.receipt.json"})
        for fold in range(3):
            feature_stem = f"partitions/features/m38b.{identity}.fold{fold}"
            paths.update({
                f"{feature_stem}.fit.features.npz",
                f"{feature_stem}.score.features.npz",
                f"{feature_stem}.features.receipt.json",
            })
        families = ("analytic", "tcn") if identity in {"RE", "SHAM"} else ("tcn",)
        for family in families:
            seeds = (1103,) if family == "analytic" else (1103, 2207, 3301)
            for fold in range(3):
                for seed in seeds:
                    stem = f"predictions/folds/m38b.{family}.{identity}.fold{fold}.seed{seed}"
                    paths.update({f"{stem}.prediction.npz", f"{stem}.prediction.receipt.json"})
                    if family == "tcn":
                        paths.add(f"{stem}.checkpoint.pt")
            oof = f"predictions/oof/m38b.{family}.{identity}.oof"
            paths.update({f"{oof}.npz", f"{oof}.receipt.json"})
    for fold in range(3):
        stem = f"partitions/truth/m38b.fold{fold}"
        paths.update({
            f"{stem}.fit.truth.npz",
            f"{stem}.score.truth.npz",
            f"{stem}.truth.receipt.json",
        })
    require(len(paths) == EXPECTED_PUBLISHED_FILES,
            "internal M38B publication inventory is inconsistent")
    return paths


def verify_inventory(run_dir: Path, ledger: HashLedger) -> dict[str, Any]:
    expected = expected_relative_paths()
    observed = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*") if path.is_file()
    }
    missing, unexpected = sorted(expected - observed), sorted(observed - expected)
    require(not missing, f"M38B publication is incomplete; missing: {missing[:5]}")
    require(not unexpected, f"M38B publication has unexpected files: {unexpected[:5]}")
    for relative in sorted(observed):
        ledger.audit(run_dir / relative)
    return {"expected": len(expected), "observed": len(observed), "exact": True}


def _source_path(source_root: Path, name: str) -> Path:
    candidates = [
        source_root / "bin" / name,
        source_root / "conf" / name,
        source_root / "modules" / name,
        source_root / "workflows" / name,
    ]
    matches = [path for path in candidates if path.is_file()]
    require(len(matches) == 1, f"load-bearing source cannot be resolved uniquely: {name}")
    return matches[0]


def verify_contract(run_dir: Path, source_root: Path, ledger: HashLedger,
                    verify_sources: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    path = run_dir / "contract/m38b.model_contract.receipt.json"
    ledger.audit(path)
    document = strict_json_load(path)
    require(sha256_file(path) == EXPECTED_MODEL_CONTRACT_RECEIPT_SHA256,
            "M38B authenticated model-contract receipt hash differs")
    require(document.get("stage") == "M38B_AUTHENTICATE_MODEL_CONTRACT"
            and document.get("status") == "PASS_BASE_AND_PRE_OUTCOME_AMENDMENT_BOUND",
            "M38B model-contract receipt status differs")
    require(document.get("base_contract_sha256") == EXPECTED_BASE_CONTRACT_SHA256,
            "M38B base contract hash differs")
    require(document.get("amendment_sha256") == EXPECTED_AMENDMENT_SHA256,
            "M38B amendment 1 hash differs")
    require(document.get("amendment_2_sha256") == EXPECTED_AMENDMENT_2_SHA256,
            "M38B amendment 2 hash differs")
    scope = document.get("scope", {})
    require(scope == {"chromosome": "22", "root": "R0", "partition": "FIT",
                      "people": 96, "valid_opened": False, "test_opened": False}
            and type(scope.get("people")) is int
            and type(scope.get("valid_opened")) is bool
            and type(scope.get("test_opened")) is bool,
            "M38B authenticated scope differs")
    manifest = document.get("source_manifest")
    require(isinstance(manifest, list) and len(manifest) == 28,
            "M38B load-bearing source manifest must contain 28 files")
    require(document.get("source_manifest_sha256") == _canonical_json_sha256(manifest),
            "M38B load-bearing source-manifest hash differs")
    if verify_sources:
        names: set[str] = set()
        for row in manifest:
            require(isinstance(row, dict) and set(row) == {"name", "sha256", "bytes"},
                    "M38B source-manifest row differs")
            require(type(row["bytes"]) is int and row["bytes"] >= 0,
                    "M38B source-manifest byte count must be a non-negative integer")
            name = row["name"]
            require(isinstance(name, str) and name not in names,
                    "M38B source-manifest names must be unique")
            names.add(name)
            source = _source_path(source_root, name)
            require(source.stat().st_size == row["bytes"]
                    and sha256_file(source) == require_hash(row["sha256"], f"source {name}"),
                    f"load-bearing source changed after M38B authentication: {name}")
        require(names == set(LOAD_BEARING_SOURCE_NAMES),
                "M38B load-bearing source-name set differs")
    provenance = {
        "model_contract_receipt_sha256": sha256_file(path),
        "base_contract_sha256": EXPECTED_BASE_CONTRACT_SHA256,
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "amendment_2_sha256": EXPECTED_AMENDMENT_2_SHA256,
    }
    return document, provenance


def verify_prelaunch(run_dir: Path, contract: Mapping[str, Any],
                     ledger: HashLedger) -> dict[str, Any]:
    prelaunch_contract_path = run_dir / "prelaunch/m38b.model_contract.receipt.json"
    published_contract_path = run_dir / "contract/m38b.model_contract.receipt.json"
    provenance_path = run_dir / "prelaunch/run_provenance.receipt.json"
    prelaunch_hash = ledger.audit(prelaunch_contract_path)
    published_hash = ledger.audit(published_contract_path)
    require(prelaunch_hash == published_hash
            and prelaunch_contract_path.read_bytes() == published_contract_path.read_bytes(),
            "prelaunch and published model-contract receipts are not byte-identical")
    prelaunch_contract = strict_json_load(prelaunch_contract_path)
    _same_json(dict(contract), prelaunch_contract, "prelaunch.contract")
    provenance_hash = ledger.audit(provenance_path)
    require(provenance_hash == EXPECTED_RUN_PROVENANCE_SHA256,
            "M38B prelaunch run-provenance receipt hash differs")
    provenance = strict_json_load(provenance_path)
    require(provenance.get("schema_version") == "1.0.0"
            and provenance.get("stage") == "M38B_PRELAUNCH_SOURCE_BINDING"
            and provenance.get("status") == "PASS_CLEAN_COMMIT_BOUND_TO_SOURCE_MANIFEST"
            and provenance.get("run_id") == run_dir.name
            and provenance.get("git_branch") == "hpc"
            and provenance.get("git_commit") == EXPECTED_GIT_COMMIT,
            "M38B prelaunch run/source identity differs")
    require_exact_bool(provenance.get("git_worktree_clean"),
                       "prelaunch.git_worktree_clean")
    require(provenance["git_worktree_clean"] is True,
            "M38B prelaunch was not launched from a clean worktree")
    require(provenance.get("origin_push_status") ==
            "PENDING_ENVIRONMENT_SECURITY_APPROVAL",
            "M38B prelaunch origin-push status differs")
    require(provenance.get("model_contract_receipt_sha256") == prelaunch_hash
            and provenance.get("source_manifest_sha256") ==
                contract.get("source_manifest_sha256")
            and type(provenance.get("source_manifest_entries")) is int
            and provenance.get("source_manifest_entries") == 28
            and provenance.get("base_contract_sha256") == EXPECTED_BASE_CONTRACT_SHA256
            and provenance.get("amendment_1_sha256") == EXPECTED_AMENDMENT_SHA256
            and provenance.get("amendment_2_sha256") == EXPECTED_AMENDMENT_2_SHA256,
            "M38B prelaunch hash binding differs")
    scope = provenance.get("scope")
    require(scope == {"chromosome": "22", "root": "R0", "partition": "FIT",
                      "valid_opened": False, "test_opened": False}
            and type(scope.get("valid_opened")) is bool
            and type(scope.get("test_opened")) is bool,
            "M38B prelaunch scope differs")
    require(isinstance(provenance.get("created_at_utc"), str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                             provenance["created_at_utc"]) is not None,
            "M38B prelaunch timestamp differs")
    return provenance


def validate_fixed_bootstrap(document: Mapping[str, Any], receipt: Mapping[str, Any] | None,
                             label: str) -> None:
    bootstrap = document.get("bootstrap")
    require(isinstance(bootstrap, dict), f"{label} bootstrap block is absent")
    require_exact_int(bootstrap.get("replicates"), EXPECTED_BOOTSTRAP_REPLICATES,
                      f"{label} bootstrap replicates")
    require_exact_int(bootstrap.get("seed"), EXPECTED_BOOTSTRAP_SEED,
                      f"{label} bootstrap seed")
    require(bootstrap.get("unit") == "whole person"
            and bootstrap.get("stratified_by") == "outer fold",
            f"{label} bootstrap sampling unit differs")
    if receipt is not None:
        require_exact_int(receipt.get("bootstrap_replicates"),
                          EXPECTED_BOOTSTRAP_REPLICATES,
                          f"{label} receipt bootstrap replicates")
        require_exact_int(receipt.get("bootstrap_seed"), EXPECTED_BOOTSTRAP_SEED,
                          f"{label} receipt bootstrap seed")


def _same_json(expected: Any, observed: Any, name: str) -> None:
    if type(expected) is not type(observed):
        raise M38BPostVerificationError(
            f"{name} JSON type differs: {type(observed).__name__} != {type(expected).__name__}"
        )
    if isinstance(expected, dict):
        require(set(expected) == set(observed), f"{name} JSON keys differ")
        for key in expected:
            _same_json(expected[key], observed[key], f"{name}.{key}")
    elif isinstance(expected, list):
        require(len(expected) == len(observed), f"{name} JSON list length differs")
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            _same_json(left, right, f"{name}[{index}]")
    elif isinstance(expected, float):
        require(math.isfinite(expected) and math.isfinite(observed)
                and expected == observed,
                f"{name} numeric value differs")
    else:
        require(expected == observed, f"{name} value differs")


def _same_npz(expected: Mapping[str, np.ndarray], path: Path, label: str) -> None:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(expected), f"{label} NPZ members differ")
        for name, wanted in expected.items():
            observed = np.asarray(archive[name])
            wanted_array = np.asarray(wanted)
            require(observed.shape == wanted_array.shape and observed.dtype == wanted_array.dtype,
                    f"{label}.{name} shape or dtype differs")
            if observed.dtype.kind in {"f", "c"}:
                values_equal = np.array_equal(observed, wanted_array, equal_nan=True)
            else:
                values_equal = np.array_equal(observed, wanted_array)
            require(values_equal,
                    f"{label}.{name} values differ")


def _provenance(document: Mapping[str, Any], label: str) -> dict[str, str]:
    return {name: require_hash(document.get(name), f"{label}.{name}")
            for name in PROVENANCE_FIELDS}


def _interval(row: Mapping[str, Any], metric: str, label: str) -> tuple[float, float]:
    intervals = row.get("metric_delta_percentile_ci95")
    require(isinstance(intervals, dict), f"{label} metric intervals are absent")
    values = intervals.get(metric)
    require(isinstance(values, list) and len(values) == 2,
            f"{label}.{metric} interval is not evaluable")
    return (require_finite_number(values[0], f"{label}.{metric}.lower"),
            require_finite_number(values[1], f"{label}.{metric}.upper"))


def _fold_values(row: Mapping[str, Any], field_name: str, label: str) -> dict[str, float]:
    values = row.get(field_name)
    require(isinstance(values, dict) and set(values) == set(EXPECTED_FOLDS),
            f"{label}.{field_name} must contain folds 0, 1, and 2")
    return {fold: require_finite_number(values[fold], f"{label}.{field_name}.{fold}")
            for fold in EXPECTED_FOLDS}


def _no_clear_harm_from_numbers(row: Mapping[str, Any], label: str) -> bool:
    n_eff = row.get("metric_fold_n_eff")
    require(isinstance(n_eff, dict), f"{label} effective sample sizes are absent")
    for metric in ERROR_VETO_METRICS + ("boundary_f1_0_2cm",):
        counts = n_eff.get(metric)
        require(isinstance(counts, dict) and set(counts) == set(EXPECTED_FOLDS)
                and all(type(counts[fold]) is int and counts[fold] > 0
                        for fold in EXPECTED_FOLDS),
                f"{label}.{metric} is not evaluable in all folds")
    error_ok = all(_interval(row, metric, label)[0] <= 0
                   for metric in ERROR_VETO_METRICS)
    boundary_ok = _interval(row, "boundary_f1_0_2cm", label)[1] >= 0
    return error_ok and boundary_ok


def derive_family_gates(result: Mapping[str, Any]) -> dict[str, Any]:
    """Derive every gate from numeric contrasts, never from stored booleans."""
    contrasts = result.get("contrasts")
    require(isinstance(contrasts, dict)
            and set(contrasts) == {name for name, _, _ in EXPECTED_CONTRASTS},
            "M38B family contrast set differs")
    candidate_rows: dict[str, bool] = {}
    for name in ("RE-RD", "RE-SHAM"):
        row = contrasts[name]
        require(isinstance(row, dict), f"{name} contrast must be an object")
        folds = _fold_values(row, "fold_mean_deltas", name)
        upper = require_finite_number(row.get("one_sided_upper_97_5_two_family"),
                                      f"{name}.one_sided_upper_97_5_two_family")
        candidate_rows[name] = all(value < 0 for value in folds.values()) and upper < 0
    uniform_ok = True
    for name in ("RE-RD", "RE-SHAM"):
        row = contrasts[name]
        metrics = row.get("metric_deltas_left_minus_right")
        fold_metrics = row.get("metric_fold_mean_deltas")
        require(isinstance(metrics, dict) and isinstance(fold_metrics, dict)
                and isinstance(fold_metrics.get("log_loss_uniform"), dict),
                f"{name} uniform log-loss metrics are absent")
        global_delta = require_finite_number(metrics.get("log_loss_uniform"),
                                             f"{name}.log_loss_uniform")
        folds = _fold_values({"uniform": fold_metrics["log_loss_uniform"]}, "uniform", name)
        uniform_ok = uniform_ok and global_delta <= 0 and all(value <= 0 for value in folds.values())
    no_harm = _no_clear_harm_from_numbers(contrasts["RE-RD"], "RE-RD")
    no_harm_full = _no_clear_harm_from_numbers(contrasts["RE-full"], "RE-full")
    deploy_row = contrasts["RE-full"]
    require(isinstance(deploy_row, dict), "RE-full contrast must be an object")
    deploy_folds = _fold_values(deploy_row, "fold_mean_deltas", "RE-full")
    deploy_metrics = deploy_row.get("metric_deltas_left_minus_right")
    deploy_fold_metrics = deploy_row.get("metric_fold_mean_deltas")
    require(isinstance(deploy_metrics, dict) and isinstance(deploy_fold_metrics, dict)
            and isinstance(deploy_fold_metrics.get("log_loss_uniform"), dict),
            "RE-full uniform log-loss metrics are absent")
    deploy_uniform_folds = _fold_values(
        {"uniform": deploy_fold_metrics["log_loss_uniform"]}, "uniform", "RE-full",
    )
    deploy_primary = (
        all(value < 0 for value in deploy_folds.values())
        and require_finite_number(deploy_row.get("one_sided_upper_97_5_two_family"),
                                  "RE-full.one_sided_upper_97_5_two_family") < 0
        and require_finite_number(deploy_metrics.get("log_loss_uniform"),
                                  "RE-full.log_loss_uniform") <= 0
        and all(value <= 0 for value in deploy_uniform_folds.values())
    )
    return {
        "candidate_contrasts": candidate_rows,
        "candidate_incremental_gate": all(candidate_rows.values()),
        "weighted_uniform_no_sign_reversal": uniform_ok,
        "no_statistically_clear_harm": no_harm,
        "no_statistically_clear_harm_vs_full": no_harm_full,
        "deploy_improvement_over_full_flare": deploy_primary and no_harm_full,
    }


def validate_reported_family_gates(result: Mapping[str, Any], derived: Mapping[str, Any],
                                   label: str) -> None:
    contrasts = result["contrasts"]
    for name, expected in derived["candidate_contrasts"].items():
        row = contrasts[name]
        require_exact_bool(row.get("candidate_contrast_gate"),
                           f"{label}.{name}.candidate_contrast_gate")
        require(row["candidate_contrast_gate"] == expected,
                f"{label}.{name} candidate gate disagrees with numeric inputs")
        fold_values = _fold_values(row, "fold_mean_deltas", f"{label}.{name}")
        expected_count = sum(value < 0 for value in fold_values.values())
        require(type(row.get("negative_direction_folds")) is int
                and row["negative_direction_folds"] == expected_count,
                f"{label}.{name} favorable-fold count differs")
        require_exact_bool(row.get("direction_3_of_3"),
                           f"{label}.{name}.direction_3_of_3")
        require(row["direction_3_of_3"] == (expected_count == 3),
                f"{label}.{name} fold-direction gate differs")
    candidate = result.get("candidate_incremental_gate")
    secondary = result.get("secondary_gates")
    require(isinstance(candidate, dict) and isinstance(secondary, dict),
            f"{label} reported gate blocks are absent")
    reported = {
        "candidate_incremental_gate": candidate.get("pass"),
        "weighted_uniform_no_sign_reversal": secondary.get(
            "weighted_uniform_no_sign_reversal", {}).get("pass"),
        "no_statistically_clear_harm": secondary.get(
            "no_statistically_clear_harm", {}).get("pass"),
        "no_statistically_clear_harm_vs_full": secondary.get(
            "no_statistically_clear_harm_vs_full", {}).get("pass"),
        "deploy_improvement_over_full_flare": secondary.get(
            "deploy_improvement_over_full_flare", {}).get("pass"),
    }
    for name, value in reported.items():
        require_exact_bool(value, f"{label}.{name}")
        require(value == derived[name], f"{label}.{name} disagrees with numeric inputs")


def derive_positive_capacity(result: Mapping[str, Any]) -> bool:
    logical_ids = result.get("logical_ids")
    require(isinstance(logical_ids, list)
            and logical_ids == list(EXPECTED_POSITIVE_IDS)
            and len(set(logical_ids)) == len(logical_ids),
            "positive-control IDs must be exact, ordered, and unique")
    contrasts = result.get("contrasts")
    expected_ids = set(EXPECTED_POSITIVE_IDS[1:])
    require(isinstance(contrasts, dict) and set(contrasts) == expected_ids,
            "positive-control contrast grid differs")
    gates: list[bool] = []
    for logical_id in EXPECTED_POSITIVE_IDS[1:]:
        row = contrasts[logical_id]
        require(isinstance(row, dict), f"{logical_id} contrast must be an object")
        folds = _fold_values(row, "fold_mean_deltas", logical_id)
        upper = require_finite_number(row.get("one_sided_upper_98_75"),
                                      f"{logical_id}.one_sided_upper_98_75")
        passed = all(value < 0 for value in folds.values()) and upper < 0
        require_exact_bool(row.get("bonferroni_four_delta_gate"),
                           f"{logical_id}.bonferroni_four_delta_gate")
        require(row["bonferroni_four_delta_gate"] == passed,
                f"{logical_id} capacity gate disagrees with numeric inputs")
        require(type(row.get("favorable_folds")) is int
                and row["favorable_folds"] == sum(value < 0 for value in folds.values()),
                f"{logical_id} favorable-fold count differs")
        gates.append(passed)
    capacity = any(gates)
    block = result.get("capacity_gate")
    require(isinstance(block, dict), "positive capacity-gate block is absent")
    require_exact_bool(block.get("pass"), "positive.capacity_gate.pass")
    require(block["pass"] == capacity,
            "positive capacity gate disagrees with numeric contrasts")
    return capacity


def _family_paths(run_dir: Path, family: str) -> tuple[dict[str, Path], dict[str, Path]]:
    baseline = run_dir / "predictions/baselines/m38b_packed_baselines"
    predictions = {
        "full": baseline / "m38b_full.oof.npz",
        "minus": baseline / "m38b_minus.oof.npz",
        "RD": baseline / "m38b_RD.oof.npz",
        "RE": run_dir / f"predictions/oof/m38b.{family}.RE.oof.npz",
        "SHAM": run_dir / f"predictions/oof/m38b.{family}.SHAM.oof.npz",
    }
    receipts = {
        "full": baseline / "m38b_baselines.receipt.json",
        "minus": baseline / "m38b_baselines.receipt.json",
        "RD": baseline / "m38b_baselines.receipt.json",
        "RE": run_dir / f"predictions/oof/m38b.{family}.RE.oof.receipt.json",
        "SHAM": run_dir / f"predictions/oof/m38b.{family}.SHAM.oof.receipt.json",
    }
    return predictions, receipts


def _scalar_text(array: np.ndarray, label: str) -> str:
    values = np.asarray(array).reshape(-1)
    require(values.size == 1, f"{label} must be scalar")
    value = values[0]
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def verify_prediction_partition_binding(
    run_dir: Path,
    identity: str,
    fold: int,
    prediction_receipt: Mapping[str, Any],
    ledger: HashLedger,
) -> None:
    """Bind one prediction receipt to its exact truth-blind fold inputs.

    SCORE truth is deliberately absent: fitting may read only FIT truth.  The
    prediction receipt must instead identify the feature and truth partition
    receipts byte-for-byte; those receipts, in turn, must identify the exact
    partition artifacts used by the fit.
    """
    is_positive = identity in EXPECTED_POSITIVE_IDS
    expected_arm = "POSITIVE" if is_positive else identity
    require(expected_arm in {"RE", "SHAM", "POSITIVE"},
            f"unsupported partition-binding identity: {identity}")

    feature_stem = run_dir / f"partitions/features/m38b.{identity}.fold{fold}"
    fit_features = Path(f"{feature_stem}.fit.features.npz")
    score_features = Path(f"{feature_stem}.score.features.npz")
    feature_receipt_path = Path(f"{feature_stem}.features.receipt.json")
    truth_stem = run_dir / f"partitions/truth/m38b.fold{fold}"
    fit_truth = Path(f"{truth_stem}.fit.truth.npz")
    truth_receipt_path = Path(f"{truth_stem}.truth.receipt.json")

    fit_features_hash = ledger.audit(fit_features)
    score_features_hash = ledger.audit(score_features)
    feature_receipt_hash = ledger.audit(feature_receipt_path)
    fit_truth_hash = ledger.audit(fit_truth)
    truth_receipt_hash = ledger.audit(truth_receipt_path)
    feature_receipt = strict_json_load(feature_receipt_path)
    truth_receipt = strict_json_load(truth_receipt_path)

    require(
        feature_receipt.get("stage") == "M38B_PARTITION_FEATURES"
        and feature_receipt.get("status") == "PASS_TRUTH_BLIND_FEATURE_PARTITION"
        and feature_receipt.get("fold") == fold
        and feature_receipt.get("arm") == expected_arm
        and feature_receipt.get("source_arm") == expected_arm
        and feature_receipt.get("fit_output_sha256") == fit_features_hash
        and feature_receipt.get("score_output_sha256") == score_features_hash
        and feature_receipt.get("truth_read") is False,
        f"{identity}/fold{fold} feature partition receipt differs",
    )
    require_exact_int(feature_receipt.get("fit_people"), 64,
                      f"{identity}/fold{fold}.feature_receipt.fit_people")
    require_exact_int(feature_receipt.get("score_people"), 32,
                      f"{identity}/fold{fold}.feature_receipt.score_people")
    if is_positive:
        require(
            feature_receipt.get("diagnostic_only") is True
            and feature_receipt.get("source_stage") ==
                "M38B_POSITIVE_CONTROL_MATERIALIZE"
            and feature_receipt.get("positive_control_delta") == DELTA_IDS[identity],
            f"{identity}/fold{fold} positive feature provenance differs",
        )
    else:
        require(
            feature_receipt.get("diagnostic_only") is False
            and feature_receipt.get("source_stage") == "M37_TRACE_MATERIALIZE"
            and feature_receipt.get("positive_control_delta") is None,
            f"{identity}/fold{fold} production feature provenance differs",
        )

    require(
        truth_receipt.get("stage") == "M38B_PARTITION_TRUTH"
        and truth_receipt.get("status") == "PASS_NON_SELECTING_TRUTH_PARTITION"
        and truth_receipt.get("fold") == fold
        and truth_receipt.get("fit_output_sha256") == fit_truth_hash
        and truth_receipt.get("model_selection_performed") is False,
        f"{identity}/fold{fold} FIT truth partition receipt differs",
    )
    require_exact_int(truth_receipt.get("fit_people"), 64,
                      f"{identity}/fold{fold}.truth_receipt.fit_people")
    require_exact_int(truth_receipt.get("score_people"), 32,
                      f"{identity}/fold{fold}.truth_receipt.score_people")

    inner_seed = prediction_receipt.get("inner_split_seed")
    require(
        type(inner_seed) is int
        and feature_receipt.get("inner_split_seed") == inner_seed
        and truth_receipt.get("inner_split_seed") == inner_seed,
        f"{identity}/fold{fold} inner split seed differs",
    )
    require_exact_int(prediction_receipt.get("train_people"), 48,
                      f"{identity}/fold{fold}.prediction.train_people")
    require_exact_int(prediction_receipt.get("select_people"), 16,
                      f"{identity}/fold{fold}.prediction.select_people")
    require_exact_int(prediction_receipt.get("score_people"), 32,
                      f"{identity}/fold{fold}.prediction.score_people")
    require(
        prediction_receipt.get("diagnostic_only") is is_positive,
        f"{identity}/fold{fold} prediction diagnostic identity differs",
    )
    require(
        "score_truth_input" in prediction_receipt
        and prediction_receipt["score_truth_input"] is None,
        f"{identity}/fold{fold} SCORE truth must remain inaccessible",
    )
    expected_bindings = {
        "fit_features_sha256": fit_features_hash,
        "score_features_sha256": score_features_hash,
        "feature_receipt_sha256": feature_receipt_hash,
        "fit_truth_sha256": fit_truth_hash,
        "truth_receipt_sha256": truth_receipt_hash,
    }
    for name, expected_hash in expected_bindings.items():
        require(prediction_receipt.get(name) == expected_hash,
                f"{identity}/fold{fold} {name} binding differs")


def verify_oof_derivation(run_dir: Path, family: str, identity: str,
                          provenance: Mapping[str, str], ledger: HashLedger,
                          event_hash: str | None = None,
                          mask_hash: str | None = None) -> None:
    """Rebuild one OOF cube from its published fold predictions."""
    is_positive = identity in EXPECTED_POSITIVE_IDS
    arm = "POSITIVE" if is_positive else identity
    require(family in {"analytic", "tcn"}
            and (identity in {"RE", "SHAM"} or (family == "tcn" and is_positive)),
            f"unsupported OOF identity: {family}/{identity}")
    seeds = (1103,) if family == "analytic" else (1103, 2207, 3301)
    oof_path = run_dir / f"predictions/oof/m38b.{family}.{identity}.oof.npz"
    oof_receipt_path = run_dir / f"predictions/oof/m38b.{family}.{identity}.oof.receipt.json"
    ledger.audit(oof_path)
    ledger.audit(oof_receipt_path)
    receipt = strict_json_load(oof_receipt_path)
    expected_status = ("PASS_DIAGNOSTIC_TRUTH_ENCODED_POSITIVE_CONTROL" if is_positive
                       else "PASS_EXACT_ONE_OOF_PREDICTION_PER_PERSON")
    expected_stage = ("M38B_COLLECT_DIAGNOSTIC_POSITIVE_OOF" if is_positive
                      else "M38B_COLLECT_TRUTH_BLIND_OOF")
    expected_delta = DELTA_IDS[identity] if is_positive else None
    require(receipt.get("stage") == expected_stage
            and receipt.get("status") == expected_status
            and receipt.get("family") == family
            and receipt.get("arm") == arm
            and receipt.get("positive_control_delta") == expected_delta
            and require_exact_bool(receipt.get("diagnostic_only"),
                                   f"{family}/{identity}.diagnostic_only") == is_positive
            and receipt.get("people") == EXPECTED_PEOPLE
            and receipt.get("folds") == 3
            and receipt.get("score_people_per_fold") == EXPECTED_FOLD_SIZE
            and receipt.get("seeds") == list(seeds)
            and receipt.get("person_coverage_min") == 1
            and receipt.get("person_coverage_max") == 1
            and receipt.get("truth_input") is None
            and receipt.get("truth_read") is False
            and receipt.get("selector_or_checkpoint_access") is False
            and receipt.get("state_names") == list(STATE_NAMES)
            and receipt.get("output_sha256") == ledger.audit(oof_path),
            f"{family}/{identity} OOF receipt identity differs")
    require(_provenance(receipt, f"{family}/{identity}.OOF") == dict(provenance),
            f"{family}/{identity} OOF provenance differs")
    if is_positive:
        require(receipt.get("real_event_identity_sha256") == event_hash
                and receipt.get("real_event_masks_sha256") == mask_hash,
                f"{family}/{identity} OOF event provenance differs")
    else:
        require(receipt.get("real_event_identity_sha256") is None
                and receipt.get("real_event_masks_sha256") is None,
                f"{family}/{identity} production OOF carries diagnostic provenance")
    sources = receipt.get("sources")
    require(isinstance(sources, list) and len(sources) == 3 * len(seeds),
            f"{family}/{identity} OOF source manifest differs")
    source_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in sources:
        require(isinstance(row, dict)
                and set(row) == {"fold", "seed", "prediction_sha256",
                                 "prediction_receipt_sha256"}
                and type(row.get("fold")) is int
                and type(row.get("seed")) is int,
                f"{family}/{identity} OOF source row differs")
        key = (row["fold"], row["seed"])
        require(key not in source_by_key, f"{family}/{identity} OOF source is duplicated")
        source_by_key[key] = row
    expected_keys = {(fold, seed) for fold in range(3) for seed in seeds}
    require(set(source_by_key) == expected_keys,
            f"{family}/{identity} OOF fold/seed source set differs")
    with np.load(oof_path, allow_pickle=False) as archive:
        required = {"probabilities", "sample_key_sha256", "marker_pos", "marker_cM",
                    "marker_axis_sha256", "fold_ids", "family", "arm", "state_names",
                    "seed_values"}
        if is_positive:
            required.add("positive_delta")
        require(set(archive.files) == required,
                f"{family}/{identity} OOF members differ")
        oof = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    people = oof["sample_key_sha256"]
    require(people.shape == (EXPECTED_PEOPLE,) and np.unique(people).size == EXPECTED_PEOPLE
            and _scalar_text(oof["family"], "OOF family") == family
            and _scalar_text(oof["arm"], "OOF arm") == arm
            and tuple(int(value) for value in oof["seed_values"].tolist()) == seeds
            and tuple(_scalar_text(np.asarray([value]), "state")
                      for value in oof["state_names"].tolist()) == STATE_NAMES,
            f"{family}/{identity} OOF axes or identity differ")
    if is_positive:
        require(float(np.asarray(oof["positive_delta"]).reshape(-1)[0]) == expected_delta,
                f"{family}/{identity} OOF positive delta differs")
    person_index = {value: index for index, value in enumerate(people.tolist())}
    expected_probability = np.empty_like(oof["probabilities"])
    expected_folds = np.empty(EXPECTED_PEOPLE, dtype=np.uint8)
    coverage = np.zeros(EXPECTED_PEOPLE, dtype=np.uint8)
    common_axes: tuple[np.ndarray, np.ndarray] | None = None
    for fold in range(3):
        fold_rows = []
        fold_people: np.ndarray | None = None
        for seed in seeds:
            stem = run_dir / (
                f"predictions/folds/m38b.{family}.{identity}.fold{fold}.seed{seed}"
            )
            prediction_path = Path(f"{stem}.prediction.npz")
            prediction_receipt_path = Path(f"{stem}.prediction.receipt.json")
            checkpoint_path = Path(f"{stem}.checkpoint.pt")
            prediction_hash = ledger.audit(prediction_path)
            prediction_receipt_hash = ledger.audit(prediction_receipt_path)
            source = source_by_key[(fold, seed)]
            require(source.get("prediction_sha256") == prediction_hash
                    and source.get("prediction_receipt_sha256") == prediction_receipt_hash,
                    f"{family}/{identity}/fold{fold}/seed{seed} source hash differs")
            prediction_receipt = strict_json_load(prediction_receipt_path)
            require(prediction_receipt.get("stage") == "M38B_TRAIN_AND_PREDICT_OOF"
                    and prediction_receipt.get("status") ==
                        ("PASS_DIAGNOSTIC_TRUTH_ENCODED_POSITIVE_CONTROL"
                         if is_positive else "PASS_SCORE_TRUTH_INACCESSIBLE")
                    and prediction_receipt.get("fold") == fold
                    and prediction_receipt.get("seed") == seed
                    and prediction_receipt.get("family") == family
                    and prediction_receipt.get("arm") == arm
                    and prediction_receipt.get("positive_control_delta") == expected_delta
                    and prediction_receipt.get("output_sha256") == prediction_hash
                    and prediction_receipt.get("model_contract_receipt_sha256") ==
                        provenance["model_contract_receipt_sha256"]
                    and prediction_receipt.get("base_contract_sha256") ==
                        provenance["base_contract_sha256"]
                    and prediction_receipt.get("amendment_sha256") ==
                        provenance["amendment_sha256"]
                    and prediction_receipt.get("amendment_2_sha256") ==
                        provenance["amendment_2_sha256"],
                    f"{family}/{identity}/fold{fold}/seed{seed} prediction receipt differs")
            verify_prediction_partition_binding(
                run_dir, identity, fold, prediction_receipt, ledger,
            )
            if family == "tcn":
                require(prediction_receipt.get("checkpoint_sha256") ==
                        ledger.audit(checkpoint_path),
                        f"{family}/{identity}/fold{fold}/seed{seed} checkpoint hash differs")
            else:
                require(prediction_receipt.get("checkpoint_sha256") is None,
                        f"analytic/{identity}/fold{fold} unexpectedly has a checkpoint")
            if is_positive:
                require(prediction_receipt.get("real_event_identity_sha256") == event_hash
                        and prediction_receipt.get("real_event_masks_sha256") == mask_hash,
                        f"{family}/{identity}/fold{fold}/seed{seed} event provenance differs")
            with np.load(prediction_path, allow_pickle=False) as archive:
                required_prediction = {"probabilities", "sample_key_sha256", "marker_pos",
                                       "marker_cM", "marker_axis_sha256", "fold", "family",
                                       "arm", "seed"}
                if is_positive:
                    required_prediction.add("positive_delta")
                require(set(archive.files) == required_prediction,
                        f"{family}/{identity}/fold{fold}/seed{seed} prediction members differ")
                row = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            require(_scalar_text(row["family"], "prediction family") == family
                    and _scalar_text(row["arm"], "prediction arm") == arm
                    and int(row["fold"].reshape(-1)[0]) == fold
                    and int(row["seed"].reshape(-1)[0]) == seed
                    and row["probabilities"].shape ==
                        (EXPECTED_FOLD_SIZE, len(oof["marker_pos"]), len(STATE_NAMES)),
                    f"{family}/{identity}/fold{fold}/seed{seed} prediction identity differs")
            if is_positive:
                require(float(row["positive_delta"].reshape(-1)[0]) == expected_delta,
                        f"{family}/{identity}/fold{fold}/seed{seed} delta differs")
            axes = (row["marker_pos"], row["marker_cM"])
            require(np.array_equal(axes[0], oof["marker_pos"])
                    and np.array_equal(axes[1], oof["marker_cM"]),
                    f"{family}/{identity}/fold{fold}/seed{seed} marker axes differ")
            if common_axes is None:
                common_axes = axes
            else:
                require(all(np.array_equal(left, right)
                            for left, right in zip(axes, common_axes, strict=True)),
                        f"{family}/{identity} fold marker axes differ")
            if fold_people is None:
                fold_people = row["sample_key_sha256"]
            else:
                require(np.array_equal(fold_people, row["sample_key_sha256"]),
                        f"{family}/{identity}/fold{fold} seed person axes differ")
            fold_rows.append(row["probabilities"])
        require(fold_people is not None and np.unique(fold_people).size == EXPECTED_FOLD_SIZE
                and all(value in person_index for value in fold_people.tolist()),
                f"{family}/{identity}/fold{fold} person axis differs")
        indices = np.asarray([person_index[value] for value in fold_people.tolist()], dtype=np.int64)
        averaged = np.stack(fold_rows, axis=0).mean(axis=0, dtype=np.float64).astype(np.float32)
        expected_probability[indices] = averaged
        expected_folds[indices] = fold
        coverage[indices] += 1
    require(np.all(coverage == 1)
            and np.array_equal(expected_probability, oof["probabilities"])
            and np.array_equal(expected_folds, oof["fold_ids"]),
            f"{family}/{identity} OOF cube is not the exact fold/seed aggregation")


def _verify_family(run_dir: Path, family: str, contract_provenance: Mapping[str, str],
                   ledger: HashLedger, expected_markers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    score_path = run_dir / f"score/m38b.{family}.metrics.json"
    per_person_path = run_dir / f"score/m38b.{family}.metrics.per_person.npz"
    receipt_path = run_dir / f"score/m38b.{family}.metrics.receipt.json"
    truth_path = run_dir / "score/m38b.score.truth.npz"
    truth_receipt = run_dir / "score/m38b.score.truth.receipt.json"
    predictions, prediction_receipts = _family_paths(run_dir, family)
    for path in (score_path, per_person_path, receipt_path, truth_path, truth_receipt,
                 *predictions.values(), *set(prediction_receipts.values())):
        ledger.audit(path)
    observed = strict_json_load(score_path)
    receipt = strict_json_load(receipt_path)
    for path in set(prediction_receipts.values()) | {truth_receipt}:
        strict_json_load(path)
    validate_fixed_bootstrap(observed, receipt, family)
    require(observed.get("family") == family
            and observed.get("stage") == "M38B_OOF_SCORE"
            and observed.get("status") == "PASS_SCORED",
            f"M38B {family} score identity differs")
    require_exact_int(observed.get("person_count"), EXPECTED_PEOPLE,
                      f"{family} person_count")
    require_exact_int(observed.get("marker_count"), expected_markers,
                      f"{family} marker_count")
    require(observed.get("state_order") == list(STATE_NAMES)
            and observed.get("ancestry_order") == list(ANCESTRY_NAMES),
            f"{family} state or ancestry axis differs")
    require(receipt.get("stage") == "M38B_OOF_SCORE"
            and receipt.get("status") == "PASS_SCORED"
            and receipt.get("family") == family
            and receipt.get("arms") == ["RD", "RE", "SHAM", "full", "minus"]
            and receipt.get("contrasts") == [list(value) for value in EXPECTED_CONTRASTS],
            f"{family} score receipt identity differs")
    require(receipt.get("output_sha256") == ledger.audit(score_path)
            and receipt.get("per_person_output_sha256") == ledger.audit(per_person_path),
            f"{family} score output hashes differ")
    inputs = observed.get("inputs_sha256")
    require(isinstance(inputs, dict) and isinstance(inputs.get("predictions"), dict)
            and set(inputs["predictions"]) == set(EXPECTED_ARMS),
            f"{family} score input-hash block differs")
    require(inputs.get("truth") == ledger.audit(truth_path)
            and receipt.get("truth_sha256") == inputs["truth"],
            f"{family} truth hash differs")
    for arm in EXPECTED_ARMS:
        expected_hash = ledger.audit(predictions[arm])
        receipt_predictions = receipt.get("prediction_sha256")
        require(isinstance(receipt_predictions, dict),
                f"{family} receipt prediction-hash block differs")
        require(inputs["predictions"].get(arm) == expected_hash
                and receipt_predictions.get(arm) == expected_hash,
                f"{family}/{arm} prediction hash differs")
    authenticated = observed.get("authenticated_model_contract")
    require(isinstance(authenticated, dict), f"{family} authenticated provenance is absent")
    observed_provenance = _provenance(authenticated, f"{family}.authenticated")
    receipt_provenance = _provenance(receipt, f"{family}.receipt")
    require(observed_provenance == receipt_provenance,
            f"{family} score/receipt provenance differs")
    require(all(receipt_provenance[name] == contract_provenance[name]
                for name in contract_provenance),
            f"{family} score is not bound to the authenticated contract")
    verified_provenance = verify_scoring_receipts(
        predictions, prediction_receipts, truth_path, truth_receipt, family,
    )
    require(verified_provenance == observed_provenance,
            f"{family} upstream receipts differ from score provenance")
    recalculated, arrays = analyse_files(
        predictions, truth_path, EXPECTED_CONTRASTS,
        EXPECTED_BOOTSTRAP_REPLICATES, EXPECTED_BOOTSTRAP_SEED,
        EXPECTED_PEOPLE, EXPECTED_FOLD_SIZE,
    )
    recalculated["per_person_output"] = {
        "filename": per_person_path.name,
        "sha256": ledger.audit(per_person_path),
    }
    recalculated["authenticated_model_contract"] = verified_provenance
    recalculated["family"] = family
    _same_json(recalculated, observed, f"{family}.score")
    _same_npz(arrays, per_person_path, f"{family}.per_person")
    derived = derive_family_gates(recalculated)
    validate_reported_family_gates(observed, derived, family)
    return recalculated, derived


def validate_axis_identity(paths: Sequence[Path], label: str) -> None:
    require(len(paths) > 1, f"{label} needs at least two NPZ artifacts")
    def axes(path: Path) -> tuple[np.ndarray, ...]:
        with np.load(path, allow_pickle=False) as archive:
            person_key = ("sample_key_sha256" if "sample_key_sha256" in archive.files
                          else "person_ids")
            required = {person_key, "fold_ids", "marker_pos", "marker_cM", "state_names"}
            require(required.issubset(archive.files), f"{label} axis members are absent")
            return tuple(np.ascontiguousarray(archive[name]) for name in (
                person_key, "fold_ids", "marker_pos", "marker_cM", "state_names",
            ))
    reference = axes(paths[0])
    for path in paths[1:]:
        candidate = axes(path)
        require(all(np.array_equal(left, right)
                    for left, right in zip(candidate, reference, strict=True)),
                f"{label} person/fold/position/cM axes differ")


def _verify_positive_materializations(run_dir: Path, ledger: HashLedger) -> tuple[str, str]:
    real_path = run_dir / "features/m38b.RE.trace.npz"
    real_receipt_path = run_dir / "features/m38b.RE.trace.receipt.json"
    ledger.audit(real_path)
    ledger.audit(real_receipt_path)
    real_receipt = strict_json_load(real_receipt_path)
    require(real_receipt.get("arm") == "RE"
            and real_receipt.get("output_sha256") == ledger.audit(real_path),
            "real RE feature receipt differs")
    real = load_npz(real_path)
    event_hash = _array_bundle_sha256(real, EVENT_IDENTITY_MEMBERS)
    mask_hash = _array_bundle_sha256(real, EVENT_MASK_MEMBERS)
    axis_hash = _array_bundle_sha256(real, AXIS_MEMBERS)
    real_receipt_hash = ledger.audit(real_receipt_path)
    for fold in range(3):
        for logical_id, delta in DELTA_IDS.items():
            stem = run_dir / f"controls/positive/features/m38b.{logical_id}.fold{fold}"
            feature_path = Path(f"{stem}.npz")
            receipt_path = Path(f"{stem}.receipt.json")
            ledger.audit(feature_path)
            ledger.audit(receipt_path)
            receipt = strict_json_load(receipt_path)
            require(receipt.get("stage") == "M38B_POSITIVE_CONTROL_MATERIALIZE"
                    and receipt.get("status") == "PASS_PRODUCTION_MATCHED_DIAGNOSTIC_CONTROL"
                    and receipt.get("arm") == "POSITIVE"
                    and require_exact_bool(receipt.get("diagnostic_only"),
                                           f"{logical_id}/fold{fold}.diagnostic_only")
                    and type(receipt.get("fold")) is int and receipt["fold"] == fold
                    and type(receipt.get("delta")) is float and receipt["delta"] == delta,
                    f"{logical_id}/fold{fold} materialization identity differs")
            require(receipt.get("output_sha256") == ledger.audit(feature_path)
                    and receipt.get("axis_sha256") == axis_hash
                    and receipt.get("real_event_identity_sha256") == event_hash
                    and receipt.get("real_event_masks_sha256") == mask_hash,
                    f"{logical_id}/fold{fold} is not bound to real RE geometry")
            inputs = receipt.get("inputs")
            require(isinstance(inputs, dict)
                    and inputs.get("real_features_sha256") == ledger.audit(real_path)
                    and inputs.get("real_receipt_sha256") == real_receipt_hash,
                    f"{logical_id}/fold{fold} is not bound to real RE inputs")
            payload = load_npz(feature_path)
            require(_array_bundle_sha256(payload, AXIS_MEMBERS) == axis_hash
                    and _array_bundle_sha256(payload, EVENT_IDENTITY_MEMBERS) == event_hash
                    and _array_bundle_sha256(payload, EVENT_MASK_MEMBERS) == mask_hash,
                    f"{logical_id}/fold{fold} positive geometry differs from RE")
            require(all(name in payload and np.count_nonzero(payload[name]) == 0
                        for name in POSITIVE_ZERO_CHANNELS),
                    f"{logical_id}/fold{fold} retains a real biological channel")
            require("event_values" in payload and payload["event_values"].ndim == 2
                    and payload["event_values"].shape[1] == 23
                    and np.count_nonzero(payload["event_values"][:, :4]) == 0
                    and np.count_nonzero(payload["event_values"][:, 10:]) == 0,
                    f"{logical_id}/fold{fold} event_values retains a biological channel")
            if delta == 0.0:
                require(all(name in payload and np.count_nonzero(payload[name]) == 0
                            for name in ("evidence_field", "event_values", "event_loglik")),
                        f"{logical_id}/fold{fold} delta-zero control contains model signal")
    return event_hash, mask_hash


def _recompute_positive(run_dir: Path, contract_provenance: Mapping[str, str],
                        ledger: HashLedger, event_hash: str,
                        mask_hash: str) -> tuple[dict[str, Any], bool]:
    score_path = run_dir / "controls/positive/score/m38b.positive.metrics.json"
    receipt_path = run_dir / "controls/positive/score/m38b.positive.metrics.receipt.json"
    truth_path = run_dir / "score/m38b.score.truth.npz"
    truth_receipt_path = run_dir / "score/m38b.score.truth.receipt.json"
    for path in (score_path, receipt_path, truth_path, truth_receipt_path):
        ledger.audit(path)
    observed = strict_json_load(score_path)
    receipt = strict_json_load(receipt_path)
    truth_receipt = strict_json_load(truth_receipt_path)
    validate_fixed_bootstrap(observed, None, "positive")
    require(observed.get("stage") == "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL"
            and observed.get("status") == "PASS_DIAGNOSTIC_GRID_SCORED"
            and observed.get("family") == "tcn"
            and require_exact_bool(observed.get("diagnostic_only"),
                                   "positive.diagnostic_only"),
            "positive score identity differs")
    require(receipt.get("stage") == observed["stage"]
            and receipt.get("status") == observed["status"]
            and receipt.get("family") == "tcn"
            and receipt.get("logical_ids") == list(EXPECTED_POSITIVE_IDS)
            and receipt.get("output_sha256") == ledger.audit(score_path),
            "positive score receipt differs")
    observed_provenance = observed.get("model_contract_provenance")
    require(isinstance(observed_provenance, dict), "positive score provenance is absent")
    observed_provenance = _provenance(observed_provenance, "positive.score")
    receipt_provenance = _provenance(receipt, "positive.receipt")
    require(observed_provenance == receipt_provenance,
            "positive score/receipt provenance differs")
    require(all(receipt_provenance[name] == contract_provenance[name]
                for name in contract_provenance),
            "positive score is not bound to the authenticated contract")
    truth_provenance = _provenance(truth_receipt, "score.truth.receipt")
    require(truth_provenance == observed_provenance
            and truth_receipt.get("output_sha256") == ledger.audit(truth_path),
            "positive truth/provenance binding differs")
    truth, people, folds, marker_cm, marker_pos = load_truth(truth_path)
    require(len(people) == EXPECTED_PEOPLE and len(marker_cm) == EXPECTED_MARKERS,
            "positive-control truth dimensions differ")
    fold_values = sorted(set(folds.tolist()), key=str)
    require(fold_values == list(EXPECTED_FOLDS)
            and all(np.count_nonzero(folds == value) == EXPECTED_FOLD_SIZE
                    for value in fold_values),
            "positive-control fold axis differs")
    scores = {}
    positive_provenance_rows = []
    positive_paths: list[Path] = []
    for logical_id in EXPECTED_POSITIVE_IDS:
        prediction = run_dir / f"predictions/oof/m38b.tcn.{logical_id}.oof.npz"
        prediction_receipt = run_dir / f"predictions/oof/m38b.tcn.{logical_id}.oof.receipt.json"
        ledger.audit(prediction)
        ledger.audit(prediction_receipt)
        strict_json_load(prediction_receipt)
        probability, provenance = load_positive(
            prediction, prediction_receipt, logical_id,
            (people, folds, marker_cm, marker_pos),
        )
        document = strict_json_load(prediction_receipt)
        require(document.get("real_event_identity_sha256") == event_hash
                and document.get("real_event_masks_sha256") == mask_hash,
                f"{logical_id} OOF is not bound to real RE event geometry")
        require(document.get("output_sha256") == ledger.audit(prediction),
                f"{logical_id} OOF hash differs")
        positive_provenance_rows.append(provenance)
        scores[logical_id] = score_arm(probability, truth, marker_cm)
        positive_paths.append(prediction)
    validate_axis_identity(positive_paths, "positive OOF grid")
    core_rows = [row[:6] for row in positive_provenance_rows]
    require(len(set(core_rows + [tuple(observed_provenance[name]
                                      for name in PROVENANCE_FIELDS)])) == 1,
            "positive OOF inputs do not share score provenance")
    require(len({row[6:] for row in positive_provenance_rows}) == 1
            and positive_provenance_rows[0][6:] == (event_hash, mask_hash),
            "positive OOF grid does not share real RE event identity")
    indices = stratified_person_bootstrap_indices(
        folds, EXPECTED_BOOTSTRAP_REPLICATES, EXPECTED_BOOTSTRAP_SEED,
    )
    zero = scores["POS_d0"].per_person["log_loss_cm"]
    contrasts: dict[str, dict[str, Any]] = {}
    for logical_id in EXPECTED_POSITIVE_IDS[1:]:
        delta_values = scores[logical_id].per_person["log_loss_cm"] - zero
        mean_delta = float(delta_values.mean())
        bootstrap = delta_values[indices].mean(axis=1)
        upper = mean_delta + float(np.quantile(mean_delta - bootstrap, 0.9875))
        fold_means = {
            str(fold): float(delta_values[folds == fold].mean())
            for fold in fold_values
        }
        passed = all(value < 0 for value in fold_means.values()) and upper < 0
        contrasts[logical_id] = {
            "delta": DELTA_IDS[logical_id],
            "contrast": f"{logical_id}-POS_d0",
            "mean_log_loss_cm_delta": mean_delta,
            "fold_mean_deltas": fold_means,
            "favorable_folds": sum(value < 0 for value in fold_means.values()),
            "one_sided_upper_98_75": upper,
            "bonferroni_four_delta_gate": passed,
        }
    recalculated = {
        "schema_version": "1.0.0",
        "stage": "M38B_SCORE_DIAGNOSTIC_POSITIVE_CONTROL",
        "status": "PASS_DIAGNOSTIC_GRID_SCORED",
        "diagnostic_only": True,
        "family": "tcn",
        "logical_ids": list(EXPECTED_POSITIVE_IDS),
        "comparison": "POS_DELTA_MINUS_POS_ZERO",
        "contrasts": contrasts,
        "capacity_gate": {
            "pass": any(row["bonferroni_four_delta_gate"] for row in contrasts.values()),
            "rule": "at least one delta: favorable in 3/3 folds and one-sided 98.75% upper bound below zero",
        },
        "bootstrap": {
            "replicates": EXPECTED_BOOTSTRAP_REPLICATES,
            "seed": EXPECTED_BOOTSTRAP_SEED,
            "unit": "whole person",
            "stratified_by": "outer fold",
        },
        "model_contract_provenance": observed_provenance,
    }
    _same_json(recalculated, observed, "positive.score")
    capacity = derive_positive_capacity(observed)
    return recalculated, capacity


def derive_final_document(family_gates: Mapping[str, Mapping[str, Any]], capacity: bool,
                          provenance: Mapping[str, str]) -> dict[str, Any]:
    require_exact_bool(capacity, "positive capacity")
    decisions: dict[str, dict[str, Any]] = {}
    for family in ("analytic", "tcn"):
        gates = family_gates[family]
        for name in (
            "candidate_incremental_gate", "weighted_uniform_no_sign_reversal",
            "no_statistically_clear_harm", "no_statistically_clear_harm_vs_full",
            "deploy_improvement_over_full_flare",
        ):
            require_exact_bool(gates.get(name), f"{family}.{name}")
        family_capacity = True if family == "analytic" else capacity
        supported = bool(
            gates["candidate_incremental_gate"] and family_capacity
            and gates["weighted_uniform_no_sign_reversal"]
            and gates["no_statistically_clear_harm"]
        )
        deploy = bool(
            supported and gates["deploy_improvement_over_full_flare"]
            and gates["no_statistically_clear_harm_vs_full"]
        )
        decisions[family] = {
            "incremental_information_supported": supported,
            "improvement_over_full_flare_supported": deploy,
            "capacity_gate": (
                "NOT_APPLICABLE_EXPLICIT_ANALYTIC_TRANSFORM"
                if family == "analytic" else family_capacity
            ),
            "family_selected": False,
            "status": (
                "SUPPORTED" if supported else
                "CAPACITY_INCONCLUSIVE" if family == "tcn" and not family_capacity
                else "NOT_SUPPORTED"
            ),
        }
    return {
        "schema_version": "1.0.0",
        "stage": "M38B_FINAL_PRESPECIFIED_DECISION",
        "status": "PASS_GATES_EVALUATED_NO_FAMILY_SELECTION",
        "families": decisions,
        "any_incremental_candidate_supported": any(
            row["incremental_information_supported"] for row in decisions.values()
        ),
        "any_improvement_over_full_flare_supported": any(
            row["improvement_over_full_flare_supported"] for row in decisions.values()
        ),
        "full_minus_scope": "ALL_S660_LOCI",
        "trace_tcn_scope": "S_STAR_123_LOO_STABLE_LOCI",
        "claim_scope": "EXPLORATORY_CHR22_R0_FIT_DONOR_CONDITIONAL",
        "provenance": dict(provenance),
    }


def _verify_final(run_dir: Path, family_scores: Mapping[str, Mapping[str, Any]],
                  family_gates: Mapping[str, Mapping[str, Any]], capacity: bool,
                  provenance: Mapping[str, str], ledger: HashLedger) -> dict[str, Any]:
    decision_path = run_dir / "decision/m38b.final_decision.json"
    receipt_path = run_dir / "decision/m38b.final_decision.receipt.json"
    positive_path = run_dir / "controls/positive/score/m38b.positive.metrics.json"
    positive_receipt = run_dir / "controls/positive/score/m38b.positive.metrics.receipt.json"
    score_paths = {
        family: run_dir / f"score/m38b.{family}.metrics.json"
        for family in ("analytic", "tcn")
    }
    score_receipts = {
        family: run_dir / f"score/m38b.{family}.metrics.receipt.json"
        for family in ("analytic", "tcn")
    }
    for path in (decision_path, receipt_path, positive_path, positive_receipt,
                 *score_paths.values(), *score_receipts.values()):
        ledger.audit(path)
    observed = strict_json_load(decision_path)
    receipt = strict_json_load(receipt_path)
    expected = derive_final_document(family_gates, capacity, provenance)
    _same_json(expected, observed, "final.decision")
    require(receipt.get("stage") == expected["stage"]
            and receipt.get("status") == expected["status"]
            and receipt.get("output_sha256") == ledger.audit(decision_path)
            and _provenance(receipt, "final.receipt") == dict(provenance),
            "final-decision receipt differs")
    # Bind direct decision inputs here because the historical decision receipt
    # only retained their shared upstream provenance.
    direct_inputs = {
        "analytic_score": ledger.audit(score_paths["analytic"]),
        "analytic_score_receipt": ledger.audit(score_receipts["analytic"]),
        "tcn_score": ledger.audit(score_paths["tcn"]),
        "tcn_score_receipt": ledger.audit(score_receipts["tcn"]),
        "positive_score": ledger.audit(positive_path),
        "positive_score_receipt": ledger.audit(positive_receipt),
    }
    require(family_scores["analytic"].get("family") == "analytic"
            and family_scores["tcn"].get("family") == "tcn",
            "final-decision family score identities differ")
    return {"document": expected, "direct_input_sha256": direct_inputs}


def _verify_rd_equals_minus(run_dir: Path) -> None:
    root = run_dir / "predictions/baselines/m38b_packed_baselines"
    minus = load_prediction(root / "m38b_minus.oof.npz", "minus")
    rd = load_prediction(root / "m38b_RD.oof.npz", "RD")
    require(np.array_equal(minus.probabilities, rd.probabilities)
            and np.array_equal(minus.person_ids, rd.person_ids)
            and np.array_equal(minus.fold_ids, rd.fold_ids)
            and np.array_equal(minus.marker_cm, rd.marker_cm)
            and minus.marker_pos is not None and rd.marker_pos is not None
            and np.array_equal(minus.marker_pos, rd.marker_pos),
            "RD must be the exact untrained F-minus baseline")


def verify_post(run_dir: Path, source_root: Path, output: Path, receipt_output: Path,
                *, require_inventory_check: bool = True,
                verify_sources: bool = True,
                expected_markers: int = EXPECTED_MARKERS) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), f"M38B run directory does not exist: {run_dir}")
    require(not output.exists() and not receipt_output.exists(),
            "refusing to overwrite M38B POST verification outputs")
    ledger = HashLedger(run_dir)
    inventory = (verify_inventory(run_dir, ledger) if require_inventory_check
                 else {"expected": None, "observed": None, "exact": None})
    contract, partial_provenance = verify_contract(
        run_dir, source_root, ledger, verify_sources=verify_sources,
    )
    prelaunch = verify_prelaunch(run_dir, contract, ledger)
    truth_receipt = strict_json_load(run_dir / "score/m38b.score.truth.receipt.json")
    ledger.audit(run_dir / "score/m38b.score.truth.receipt.json")
    full_provenance = _provenance(truth_receipt, "score.truth.receipt")
    require(all(full_provenance[name] == value for name, value in partial_provenance.items()),
            "truth is not bound to the authenticated M38B contract")
    event_hash, mask_hash = _verify_positive_materializations(run_dir, ledger)
    for family in ("analytic", "tcn"):
        for identity in ("RE", "SHAM"):
            verify_oof_derivation(
                run_dir, family, identity, full_provenance, ledger,
            )
    for identity in EXPECTED_POSITIVE_IDS:
        verify_oof_derivation(
            run_dir, "tcn", identity, full_provenance, ledger,
            event_hash, mask_hash,
        )
    family_scores: dict[str, dict[str, Any]] = {}
    family_gates: dict[str, dict[str, Any]] = {}
    for family in ("analytic", "tcn"):
        family_scores[family], family_gates[family] = _verify_family(
            run_dir, family, full_provenance, ledger, expected_markers,
        )
    for arm in ("full", "minus", "RD"):
        _same_json(
            family_scores["analytic"]["arm_metrics"][arm],
            family_scores["tcn"]["arm_metrics"][arm],
            f"shared_baseline.{arm}",
        )
    _same_json(
        family_scores["analytic"]["contrasts"]["full-minus"],
        family_scores["tcn"]["contrasts"]["full-minus"],
        "shared_baseline.full-minus",
    )
    _verify_rd_equals_minus(run_dir)
    positive_score, capacity = _recompute_positive(
        run_dir, full_provenance, ledger, event_hash, mask_hash,
    )
    final = _verify_final(
        run_dir, family_scores, family_gates, capacity, full_provenance, ledger,
    )
    manifest = ledger.document()
    manifest_hash = _canonical_json_sha256(manifest)
    report = {
        "schema_version": "1.0.0",
        "stage": "M38B_POST_HOC_FAIL_CLOSED_VERIFICATION",
        "status": "PASS_ALL_SCORES_BOOTSTRAPS_GATES_AND_HASHES_RECOMPUTED",
        "training_performed": False,
        "run_directory_name": run_dir.name,
        "scope": contract["scope"],
        "prelaunch": {
            "git_commit": prelaunch["git_commit"],
            "created_at_utc": prelaunch["created_at_utc"],
            "model_contract_receipt_sha256": prelaunch[
                "model_contract_receipt_sha256"
            ],
            "run_provenance_receipt_sha256": ledger.audit(
                run_dir / "prelaunch/run_provenance.receipt.json"
            ),
        },
        "fixed_inference": {
            "bootstrap_replicates": EXPECTED_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
            "unit": "whole person stratified by outer fold",
        },
        "inventory": inventory,
        "audited_input_manifest_sha256": manifest_hash,
        "audited_input_count": len(manifest),
        "audited_inputs": manifest,
        "axes": {
            "people": EXPECTED_PEOPLE,
            "markers": expected_markers,
            "outer_folds": 3,
            "score_people_per_fold": EXPECTED_FOLD_SIZE,
            "state_order": list(STATE_NAMES),
            "ancestry_order": list(ANCESTRY_NAMES),
        },
        "positive_control_binding": {
            "logical_ids": list(EXPECTED_POSITIVE_IDS),
            "real_RE_event_identity_sha256": event_hash,
            "real_RE_event_masks_sha256": mask_hash,
            "capacity_gate_recomputed": capacity,
        },
        "family_gates_recomputed": family_gates,
        "final_decision_recomputed": final["document"],
        "final_decision_direct_input_sha256": final["direct_input_sha256"],
        "claim_scope": "EXPLORATORY_CHR22_R0_FIT_DONOR_CONDITIONAL",
        "interpretation_guardrail": (
            "A pass authenticates and recomputes this fixed experiment; it does not "
            "generalise to real DNABR, other chromosomes, roots, donor panels, or all "
            "rare-variant representations."
        ),
    }
    _write_exclusive_json(output, report)
    verifier_hash = sha256_file(Path(__file__))
    receipt = {
        "schema_version": "1.0.0",
        "stage": report["stage"],
        "status": report["status"],
        "output_sha256": sha256_file(output),
        "verifier_sha256": verifier_hash,
        "audited_input_manifest_sha256": manifest_hash,
        "audited_input_count": len(manifest),
        "contract_receipt_sha256": full_provenance["model_contract_receipt_sha256"],
        "direct_decision_inputs_sha256": final["direct_input_sha256"],
        "bootstrap_replicates": EXPECTED_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
    }
    _write_exclusive_json(receipt_output, receipt)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Local copy of the complete canonical M38B run prefix")
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="Repository root used to verify the 28 load-bearing sources")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path,
                        help="Defaults to OUTPUT with .receipt.json suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = args.receipt or args.output.with_suffix(".receipt.json")
    result = verify_post(args.run_dir, args.source_root, args.output, receipt)
    print(json.dumps({
        "status": result["status"],
        "audited_input_count": result["audited_input_count"],
        "any_incremental_candidate_supported": result[
            "final_decision_recomputed"
        ]["any_incremental_candidate_supported"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
