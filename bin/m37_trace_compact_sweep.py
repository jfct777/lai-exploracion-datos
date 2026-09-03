#!/usr/bin/env python3
"""Run authenticated M37 FIT/TUNE candidates without persisting predictions.

The compact lane reuses the canonical materialized feature packages.  It first
replays the previously executed default candidate and compares its metrics with
the authenticated canonical R0 collection.  Only after that equivalence check
passes does it evaluate the candidate manifest.  Probabilities and checkpoints
remain process-local; the persistent outputs are metrics and audit receipts.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from m37_trace_collect_metrics import ARMS
from m37_trace_core import TraceSpec, hmm_posterior, require
from m37_trace_score import boundaries, match_count, score
from m37_trace_train import (authenticate_feature_pair, authenticate_truth_axes,
                             deterministic_train_tune, load_features, load_truth, train,
                             verify_stage_receipt)


METRIC_PATHS = (
    ("log_loss",),
    ("brier",),
    ("macro_ancestry_dose_mae",),
    ("calibration_ece_15",),
    ("false_transitions_per_morgan",),
    ("mean_boundary_error_cM",),
    *(("ancestry_dose_mae", ancestry) for ancestry in ("AFR", "EUR", "NAM")),
    *(("f1_boundary", tolerance) for tolerance in ("0.05", "0.1", "0.2", "0.5")),
    *(("baseline", name) for name in (
        "log_loss", "brier", "macro_ancestry_dose_mae", "calibration_ece_15",
        "false_transitions_per_morgan", "mean_boundary_error_cM",
    )),
    *(("baseline", "ancestry_dose_mae", ancestry) for ancestry in ("AFR", "EUR", "NAM")),
    *(("baseline", "f1_boundary", tolerance) for tolerance in ("0.05", "0.1", "0.2", "0.5")),
)
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        require(isinstance(current, dict) and key in current,
                f"M37 equivalence metric lacks {'.'.join(path)}")
        current = current[key]
    return current


def compare_metrics(observed: dict[str, Any], expected: dict[str, Any],
                    absolute_tolerance: float, relative_tolerance: float
                    ) -> dict[str, dict[str, object]]:
    """Compare the complete prespecified numeric score surface."""
    require(absolute_tolerance >= 0 and relative_tolerance >= 0,
            "M37 equivalence tolerances must be nonnegative")
    comparisons: dict[str, dict[str, object]] = {}
    for path in METRIC_PATHS:
        name = ".".join(path)
        left, right = _nested(observed, path), _nested(expected, path)
        if left is None or right is None:
            passed = left is None and right is None
            delta = None
        else:
            require(isinstance(left, (int, float)) and isinstance(right, (int, float)),
                    f"M37 equivalence metric {name} is not numeric")
            left, right = float(left), float(right)
            require(math.isfinite(left) and math.isfinite(right),
                    f"M37 equivalence metric {name} is not finite")
            passed = math.isclose(left, right, abs_tol=absolute_tolerance,
                                  rel_tol=relative_tolerance)
            delta = left - right
        comparisons[name] = {
            "observed": left,
            "canonical": right,
            "difference": delta,
            "pass": bool(passed),
        }
    return comparisons


def _require_tcn_rd_raw_f0_identity(metric: dict[str, Any]) -> None:
    """Reject any RD score change caused by the residual probability operator."""
    baseline = metric.get("baseline")
    require(isinstance(baseline, dict), "M37 TCN RD metric lacks raw F0")
    for path in METRIC_PATHS:
        if path[0] == "baseline":
            continue
        observed, expected = _nested(metric, path), _nested(baseline, path)
        require(observed == expected,
                f"M37 TCN RD differs from raw F0 at {'.'.join(path)}")


def _load_contracts(parent_path: Path, amendment_path: Path,
                    manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, amendment = _json(parent_path), _json(amendment_path)
    parent_sha, amendment_sha = sha256(parent_path), sha256(amendment_path)
    binding = manifest.get("contract_binding")
    require(parent.get("stage") == "M37_TRACE_FINITE_SUCCESSIVE_HALVING" and
            parent.get("schema_version") == "1.0.0" and isinstance(binding, dict) and
            binding.get("parent_sha256") == parent_sha and
            binding.get("amendment_sha256") == amendment_sha and
            amendment.get("stage") == "M37_TRACE_COMPACT_SWEEP_AMENDMENT" and
            amendment.get("parent_contract_sha256") == parent_sha,
            "M37 compact parent/amendment contract binding differs")
    return parent, amendment


def _load_manifest(path: Path, family: str, root: str,
                   parent_contract_path: Path, amendment_path: Path
                   ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(path)
    require(manifest.get("schema_version") == "1.0.0" and
            manifest.get("stage") == "M37_TRACE_COMPACT_SWEEP_PRE" and
            manifest.get("status") == "PREREGISTERED_FIT_TUNE_ONLY",
            "M37 compact candidate manifest identity differs")
    scope = manifest.get("scope")
    require(isinstance(scope, dict) and scope.get("root") == root == "R0" and
            scope.get("evaluation_split") == "FIT_TUNE" and
            scope.get("valid_access") == "FORBIDDEN",
            "M37 compact sweep may use only R0 FIT/TUNE")
    require(tuple(manifest.get("arms", ())) == ARMS,
            "M37 compact sweep needs exactly the five paired arms")
    parent_contract, amendment = _load_contracts(parent_contract_path, amendment_path, manifest)
    execution = manifest.get("execution")
    families = manifest.get("families")
    equivalence = manifest.get("equivalence")
    candidates = manifest.get("candidates")
    positive_control_status = manifest.get("positive_control_status")
    require(isinstance(execution, dict) and isinstance(families, dict) and
            isinstance(equivalence, dict) and isinstance(candidates, list) and
            isinstance(positive_control_status, dict) and family in positive_control_status,
            "M37 compact manifest sections differ")
    capacity = amendment.get("capacity_control")
    if family == "tcn":
        declared_control = positive_control_status.get("tcn")
        require(isinstance(capacity, dict) and isinstance(declared_control, dict) and
                capacity.get("screen_seed") == declared_control.get("screen_seed") == 1103 and
                capacity.get("screen_candidate_count") ==
                declared_control.get("candidate_count") ==
                sum(row.get("family") == "tcn" for row in candidates) and
                capacity.get("candidate_evaluation") ==
                declared_control.get("candidate_evaluation") ==
                "ALL_DECLARED_CANDIDATES_ACROSS_ALL_FIXED_SEEDS_NO_RANKING" and
                capacity.get("budget_ladder_updates") ==
                declared_control.get("budget_ladder_updates") and
                capacity.get("rung_execution") ==
                declared_control.get("rung_execution") ==
                "DETERMINISTIC_RESTART_SAME_SEED_EACH_RUNG" and
                isinstance(declared_control.get("budget_ladder_updates"), list) and
                declared_control.get("budget_ladder_updates") and
                declared_control.get("budget_ladder_updates")[0] == execution.get("updates") and
                declared_control.get("effective_budget_rule") ==
                "SECOND_SMALLEST_FIRST_PASS_RUNG_SHARED_BY_ALL_FIVE_ARMS" and
                capacity.get("replication_seeds") ==
                declared_control.get("replication_seeds") == [1103, 2207, 3301] and
                declared_control.get("required_seed_passes") == 2 and
                declared_control.get("required_controls") ==
                ["additive", "xor_interaction", "xor_one_bit_ablation",
                 "zero_revival"] and
                capacity.get("valid_access") == "FORBIDDEN",
                "M37 TCN capacity-control contract differs")
    require(family in ("hmm", "tcn") and isinstance(families.get(family), dict),
            "M37 compact family differs")
    replay = equivalence.get("replay", {}).get(family)
    equivalence_policy = equivalence.get("policy_by_family", {}).get(family)
    require(equivalence_policy in {
                "REQUIRE_CANONICAL_METRIC_REPLAY",
                "NEWLY_FROZEN_RESIDUAL_OPERATOR_REFERENCE_ONLY",
            } and
            isinstance(replay, dict) and replay.get("family") == family and
            SAFE_ID.fullmatch(str(replay.get("canonical_candidate_id", ""))),
            "M37 compact default replay differs")
    require(family != "hmm" or equivalence_policy == "REQUIRE_CANONICAL_METRIC_REPLAY",
            "M37 compact HMM must retain its canonical equivalence gate")
    family_candidates = [row for row in candidates if row.get("family") == family]
    require(family_candidates, f"M37 compact manifest has no {family} candidates")
    identifiers = [str(row.get("candidate_id", "")) for row in family_candidates]
    require(all(SAFE_ID.fullmatch(value) for value in identifiers) and
            len(set(identifiers)) == len(identifiers),
            "M37 compact candidate identifiers differ")
    for row in family_candidates:
        effective = _effective_parameters(manifest, family, row)
        _validate_contract_domain(parent_contract, amendment, family, effective)
    replay_effective = _effective_parameters(manifest, family, replay)
    _validate_contract_domain(parent_contract, amendment, family, replay_effective)
    expected_count = int(amendment["candidate_count_by_family"][family])
    require(len(family_candidates) == expected_count,
            f"M37 compact {family} candidate count differs from amendment")
    effective_rows = [_effective_parameters(manifest, family, row) for row in family_candidates]
    if family == "hmm":
        observed = {(float(row["hazard_per_morgan"]), float(row["evidence_scale"]))
                    for row in effective_rows}
        expected = {(float(hazard), float(scale))
                    for hazard in amendment["hmm"]["hazard_per_morgan"]
                    for scale in amendment["hmm"]["evidence_lambda"]}
        require(observed == expected, "M37 compact HMM grid is incomplete or duplicated")
    else:
        signatures = {
            (int(row["hidden_dim"]), int(row["depth"]), int(row["kernel_size"]),
             float(row["dropout"]), tuple(map(int, row["dilations"])),
             float(row["learning_rate"]), float(row["event_radius_cM"]),
             float(row["evidence_scale"])) for row in effective_rows
        }
        require(len(signatures) == expected_count,
                "M37 compact TCN candidate signatures are duplicated")
        declared_ids = amendment["tcn"].get("candidate_ids")
        require(isinstance(declared_ids, list) and set(identifiers) == set(declared_ids),
                "M37 compact TCN candidate IDs differ from the amendment")
        declared_balance = amendment["tcn"].get("space_filling_balance")
        if declared_balance is not None:
            require(expected_count == 6 and declared_balance == {
                        "hidden_dim": {"32": 2, "64": 2, "96": 2},
                        "depth": {"2": 2, "3": 2, "4": 2},
                        "kernel_size": {"3": 3, "5": 3},
                        "dropout": {"0.0": 2, "0.1": 2, "0.2": 2},
                        "learning_rate": {"0.0001": 2, "0.0003": 2, "0.001": 2},
                        "event_radius_cM": {"0.05": 1, "0.1": 1, "0.2": 2, "0.5": 2},
                        "evidence_lambda": {"0.25": 1, "0.5": 2, "1.0": 1, "2.0": 2},
                    } and
                    Counter(int(row["hidden_dim"]) for row in effective_rows) ==
                    Counter({32: 2, 64: 2, 96: 2}) and
                    Counter(int(row["depth"]) for row in effective_rows) ==
                    Counter({2: 2, 3: 2, 4: 2}) and
                    Counter(int(row["kernel_size"]) for row in effective_rows) ==
                    Counter({3: 3, 5: 3}) and
                    Counter(float(row["dropout"]) for row in effective_rows) ==
                    Counter({0.0: 2, 0.1: 2, 0.2: 2}) and
                    Counter(float(row["learning_rate"]) for row in effective_rows) ==
                    Counter({0.0001: 2, 0.0003: 2, 0.001: 2}) and
                    Counter(float(row["event_radius_cM"]) for row in effective_rows) ==
                    Counter({0.05: 1, 0.1: 1, 0.2: 2, 0.5: 2}) and
                    Counter(float(row["evidence_scale"]) for row in effective_rows) ==
                    Counter({0.25: 1, 0.5: 2, 1.0: 1, 2.0: 2}),
                    "M37 compact TCN balanced six-row space-filling design differs")
    return manifest, replay, family_candidates


def _validate_contract_domain(parent: dict[str, Any], amendment: dict[str, Any],
                              family: str, effective: dict[str, Any]) -> None:
    numeric_values = [value for value in effective.values()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)]
    require(all(math.isfinite(float(value)) for value in numeric_values),
            "M37 compact parameters must be finite")
    triage = next((row for row in parent["rungs"] if row.get("name") == "triage"), None)
    require(isinstance(triage, dict), "M37 compact parent triage rung is absent")
    parent_execution = {
        "updates": int(triage["updates"]),
        "batch_people": int(parent["training"]["batch_people"]),
        "marker_shard": int(parent["training"]["marker_shard"]),
        "validation_every": int(parent["training"]["validation_every"]),
        "early_stopping_patience": int(parent["training"]["early_stopping_patience"]),
    }
    observed_execution = {key: int(effective[key]) for key in parent_execution}
    if family == "hmm":
        require(observed_execution == parent_execution,
                "M37 compact HMM execution differs from the parent triage contract")
    else:
        calibrated = amendment.get("tcn", {}).get("execution")
        require(isinstance(calibrated, dict),
                "M37 compact TCN calibrated execution is absent")
        ladder = calibrated.get("budget_ladder_updates")
        require(isinstance(ladder, list) and ladder and
                ladder == sorted(set(ladder)) and
                all(isinstance(value, int) and value > 0 for value in ladder) and
                observed_execution["updates"] in ladder and
                all(observed_execution[key] == int(calibrated.get(key, expected))
                    for key, expected in parent_execution.items() if key != "updates"),
                "M37 compact TCN execution is outside the calibrated budget ladder")
    if family == "hmm":
        require(float(effective["hazard_per_morgan"]) in set(map(float, parent["hmm"]["hazard_per_morgan"])) and
                float(effective["evidence_scale"]) in set(map(float, parent["hmm"]["evidence_lambda"])),
                "M37 compact HMM candidate lies outside the parent contract")
    else:
        tcn = parent["tcn"]
        require(int(effective["hidden_dim"]) in tcn["hidden_dim"] and
                int(effective["depth"]) in tcn["depth"] and
                int(effective["kernel_size"]) in tcn["kernel_size"] and
                float(effective["dropout"]) in set(map(float, tcn["dropout"])) and
                float(effective["learning_rate"]) in set(map(float, tcn["learning_rate"])) and
                int(effective["seed"]) == 1103 and
                list(map(int, effective["dilations"])) in tcn["dilations"] and
                len(effective["dilations"]) == int(effective["depth"]) and
                float(effective["event_radius_cM"]) in set(map(float, amendment["tcn"]["event_radius_cM"])) and
                float(effective["evidence_scale"]) in set(map(float, amendment["tcn"]["evidence_lambda"])),
                "M37 compact TCN candidate lies outside the bound parent/amendment domain")


def _effective_parameters(manifest: dict[str, Any], family: str,
                          candidate: dict[str, Any]) -> dict[str, Any]:
    execution = dict(manifest["execution"])
    model = dict(manifest["families"][family])
    overrides = candidate.get("parameters", {})
    require(isinstance(overrides, dict), "M37 compact candidate parameters differ")
    unexpected = set(overrides) - (set(model) | set(execution))
    require(not unexpected, f"M37 compact candidate overrides undeclared keys: {sorted(unexpected)}")
    effective = {**execution, **model, **overrides}
    required_execution = {
        "updates", "batch_people", "marker_shard", "validation_every",
        "early_stopping_patience", "tune_fraction", "split_seed", "event_radius_cM",
    }
    require(required_execution.issubset(effective), "M37 compact execution parameters differ")
    require(int(effective["updates"]) > 0 and int(effective["batch_people"]) > 0 and
            int(effective["marker_shard"]) > 0 and int(effective["validation_every"]) > 0 and
            int(effective["early_stopping_patience"]) > 0 and
            0.05 <= float(effective["tune_fraction"]) <= 0.4 and
            float(effective["event_radius_cM"]) > 0,
            "M37 compact execution values differ")
    require(float(effective.get("evidence_scale", -1)) >= 0,
            "M37 compact evidence scale must be nonnegative")
    if family == "hmm":
        require(float(effective.get("hazard_per_morgan", 0)) > 0,
                "M37 compact HMM hazard differs")
    else:
        dilations = tuple(int(value) for value in effective.get("dilations", ()))
        TraceSpec(int(effective["hidden_dim"]), int(effective["depth"]),
                  int(effective["kernel_size"]), float(effective["dropout"]), dilations)
        require(float(effective["learning_rate"]) > 0 and int(effective["seed"]) >= 0,
                "M37 compact TCN optimization values differ")
    return effective


def _load_canonical_collection(metrics_path: Path, receipt_path: Path, root: str
                               ) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    collection, receipt = _json(metrics_path), _json(receipt_path)
    observed_sha = sha256(metrics_path)
    rows = collection.get("rows")
    require(collection.get("stage") == "M37_TRACE_COLLECT_METRICS" and
            collection.get("root") == root and collection.get("evaluation_split") == "FIT_TUNE" and
            isinstance(rows, list) and rows and
            receipt.get("stage") == "M37_TRACE_COLLECT_METRICS" and
            receipt.get("root") == root and receipt.get("evaluation_split") == "FIT_TUNE" and
            receipt.get("row_count") == len(rows) and receipt.get("output_sha256") == observed_sha,
            "M37 canonical metric collection/receipt differs")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict) and row.get("root") == root and row.get("arm") in ARMS and
                isinstance(row.get("metrics"), dict), "M37 canonical metric row differs")
        key = (str(row.get("candidate_id", "")), str(row["arm"]))
        require(all(key) and key not in indexed, "M37 canonical metric identities differ")
        indexed[key] = row["metrics"]
    return indexed, receipt


def _bind_features(feature_paths: Iterable[Path], receipt_paths: Iterable[Path],
                   truth_path: Path, f0_receipt_path: Path
                   ) -> tuple[dict[str, dict[str, np.ndarray]],
                              dict[str, dict[str, Any]],
                              dict[str, dict[str, str]], dict[str, Any]]:
    paths, receipts = list(feature_paths), list(receipt_paths)
    require(len(paths) == len(receipts) == len(ARMS),
            "M37 compact sweep needs five feature artifacts and five receipts")
    artifact_sha = {sha256(path): path for path in paths}
    require(len(artifact_sha) == len(paths), "M37 compact feature artifacts are duplicated")
    bound: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for receipt_path in receipts:
        receipt = _json(receipt_path)
        arm, output_sha = str(receipt.get("arm", "")), str(receipt.get("output_sha256", ""))
        require(receipt.get("stage") == "M37_TRACE_MATERIALIZE" and arm in ARMS and
                arm not in bound and output_sha in artifact_sha and
                receipt.get("target_ref_disjoint") is True and
                receipt.get("target_fold_assignment") == "forbidden",
                "M37 compact feature receipt identity differs")
        bound[arm] = (artifact_sha[output_sha], receipt_path, receipt)
    require(set(bound) == set(ARMS), "M37 compact feature arm set differs")

    features: dict[str, dict[str, np.ndarray]] = {}
    receipt_payloads: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, str]] = {}
    anchor: dict[str, np.ndarray] | None = None
    baseline_sources: set[str] = set()
    for arm in ARMS:
        artifact, receipt_path, receipt = bound[arm]
        require(artifact.name == f"FIT.{arm}.trace.npz" and
                receipt_path.name == f"FIT.{arm}.trace.receipt.json",
                "M37 compact input names are not sealed FIT artifacts")
        loaded = load_features(artifact)
        verify_stage_receipt(artifact, receipt_path, "M37_TRACE_MATERIALIZE", arm)
        require(authenticate_feature_pair(loaded, loaded) == "FIT_TUNE",
                "M37 compact input is not a FIT/TUNE feature package")
        authenticate_truth_axes(truth_path, loaded)
        require(receipt.get("physical_genetic_axis_sha256") ==
                str(loaded["marker_axis_sha256"].reshape(-1)[0]),
                "M37 compact receipt/marker axis differs")
        baseline_source = str(loaded.get("baseline_source_sha256", np.asarray([""]))
                              .reshape(-1)[0])
        receipt_inputs = receipt.get("inputs")
        require(re.fullmatch(r"[0-9a-f]{64}", baseline_source) is not None and
                isinstance(receipt_inputs, dict) and
                receipt_inputs.get("F0_sha256") == baseline_source,
                "M37 compact feature does not bind its F0 source")
        baseline_sources.add(baseline_source)
        if anchor is None:
            anchor = loaded
        else:
            for name in ("sample_key_sha256", "marker_pos", "marker_cM",
                         "marker_axis_sha256", "state_names", "baseline_states",
                         "schedule_sample", "schedule_marker"):
                require(np.array_equal(anchor[name], loaded[name]),
                        f"M37 compact paired-arm {name} differs")
        features[arm] = loaded
        receipt_payloads[arm] = receipt
        evidence[arm] = {
            "features_sha256": sha256(artifact),
            "features_receipt_sha256": sha256(receipt_path),
        }
    require(len(baseline_sources) == 1,
            "M37 compact paired arms do not share one F0 source")
    baseline_source = next(iter(baseline_sources))
    f0_receipt = _json(f0_receipt_path)
    outputs = f0_receipt.get("outputs")
    matching_outputs = ([name for name, item in outputs.items()
                         if isinstance(item, dict) and item.get("sha256") == baseline_source]
                        if isinstance(outputs, dict) else [])
    require(f0_receipt.get("stage") == "M34_PARSE_FLARE_F0" and
            f0_receipt.get("decision") == "PASS_F0_TRUTH_BLIND" and
            f0_receipt.get("truth_opened") is False and
            f0_receipt.get("contains_truth") is False and
            f0_receipt.get("ancestry_order") == ["AFR", "EUR", "NAM"] and
            anchor is not None and
            int(f0_receipt.get("sample_count", -1)) == len(anchor["sample_key_sha256"]) and
            int(f0_receipt.get("marker_count", -1)) == len(anchor["marker_pos"]) and
            len(matching_outputs) == 1,
            "M37 compact F0 source is not bound to a truth-blind FLARE receipt")
    baseline_provenance = {
        "method": "FLARE",
        "upstream_stage": "M34_PARSE_FLARE_F0",
        "source_artifact": matching_outputs[0],
        "source_sha256": baseline_source,
        "upstream_receipt_sha256": sha256(f0_receipt_path),
        "truth_blind": True,
        "sample_count": int(f0_receipt["sample_count"]),
        "marker_count": int(f0_receipt["marker_count"]),
    }
    for loaded in features.values():
        loaded["baseline_method"] = np.asarray([baseline_provenance["method"]])
        loaded["baseline_upstream_stage"] = np.asarray(
            [baseline_provenance["upstream_stage"]],
        )
        loaded["baseline_receipt_sha256"] = np.asarray(
            [baseline_provenance["upstream_receipt_sha256"]],
        )
    return features, receipt_payloads, evidence, baseline_provenance


def _candidate_metric(probabilities: np.ndarray, tune_people: np.ndarray,
                      features: dict[str, np.ndarray], truth: np.ndarray,
                      candidate_id: str, family: str, root: str, arm: str
                      ) -> dict[str, Any]:
    tune_probabilities, tune_truth = probabilities[tune_people], truth[tune_people]
    result = score(tune_probabilities, tune_truth, features["marker_cM"])
    result["baseline"] = score(features["baseline_states"][tune_people], truth[tune_people],
                               features["marker_cM"])
    result["baseline_metadata"] = {
        "method": str(features.get("baseline_method", np.asarray(["upstream_baseline_unspecified"]))[0]),
        "source_sha256": str(features.get("baseline_source_sha256", np.asarray(["unavailable"]))[0]),
        "upstream_stage": str(features.get("baseline_upstream_stage", np.asarray(["unavailable"]))[0]),
        "upstream_receipt_sha256": str(features.get("baseline_receipt_sha256", np.asarray(["unavailable"]))[0]),
    }
    result.update({
        "evaluation_split": "FIT_TUNE",
        "candidate_id": candidate_id,
        "root": root,
        "family": family,
        "arm": arm,
        "marker_axis_sha256": str(features["marker_axis_sha256"].reshape(-1)[0]),
    })
    # Paired person-level rows support uncertainty analyses without retaining
    # the much larger probability tensor.  The key is already a one-way digest
    # from the canonical feature package; no sample identifier is introduced.
    person_rows: list[dict[str, Any]] = []
    sample_keys = np.asarray(features["sample_key_sha256"])
    for local_index, sample_index in enumerate(tune_people.tolist()):
        observed = score(tune_probabilities[local_index:local_index + 1],
                         tune_truth[local_index:local_index + 1], features["marker_cM"])
        baseline_observed = score(
            features["baseline_states"][sample_index:sample_index + 1],
            truth[sample_index:sample_index + 1], features["marker_cM"],
        )
        raw_key = sample_keys[sample_index]
        sample_key = (raw_key.decode("ascii") if isinstance(raw_key, bytes)
                      else str(raw_key))
        predicted_boundaries = boundaries(tune_probabilities[local_index].argmax(axis=1),
                                          features["marker_cM"])
        true_boundaries = boundaries(tune_truth[local_index], features["marker_cM"])
        boundary_counts: dict[str, dict[str, int]] = {}
        for tolerance in (0.05, 0.1, 0.2, 0.5):
            true_positive = match_count(true_boundaries, predicted_boundaries, tolerance)
            boundary_counts[str(tolerance)] = {
                "TP": int(true_positive),
                "FP": int(len(predicted_boundaries) - true_positive),
                "FN": int(len(true_boundaries) - true_positive),
            }
        person_rows.append({
            "sample_axis_index": int(sample_index),
            "sample_key_sha256": sample_key,
            "log_loss": observed["log_loss"],
            "brier": observed["brier"],
            "macro_ancestry_dose_mae": observed["macro_ancestry_dose_mae"],
            "ancestry_dose_mae": observed["ancestry_dose_mae"],
            "f1_boundary_0.2_cM": observed["f1_boundary"]["0.2"],
            "false_transitions_per_morgan": observed["false_transitions_per_morgan"],
            "F0_log_loss": baseline_observed["log_loss"],
            "F0_macro_ancestry_dose_mae": baseline_observed["macro_ancestry_dose_mae"],
            "F0_f1_boundary_0.2_cM": baseline_observed["f1_boundary"]["0.2"],
            "boundary_counts": boundary_counts,
        })
    result["per_individual"] = person_rows
    result["per_individual_metric_schema"] = {
        "population": "deterministic FIT-derived TUNE people only",
        "boundary_tolerance_cM": 0.2,
        "identifier": "canonical one-way sample_key_sha256",
        "boundary_counts": "TP/FP/FN from optimal one-to-one matches at each cM tolerance",
    }
    return result


def _execute_candidate(candidate_id: str, family: str, effective: dict[str, Any],
                       features_by_arm: dict[str, dict[str, np.ndarray]], truth: np.ndarray,
                       root: str) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray]:
    metrics: dict[str, dict[str, Any]] = {}
    if family == "hmm":
        # HMM rows are independent.  Concatenating the five arms turns five
        # Python traversals of 42k markers into one without changing any
        # posterior; default replay equivalence verifies this optimization.
        tune_by_arm = [deterministic_train_tune(
            features_by_arm[arm], int(effective["split_seed"]),
            float(effective["tune_fraction"]),
        ) for arm in ARMS]
        require(all(np.array_equal(tune_by_arm[0][0], row[0]) and
                    np.array_equal(tune_by_arm[0][1], row[1]) for row in tune_by_arm[1:]),
                "M37 compact HMM TRAIN/TUNE split changed between arms")
        arm_people = len(features_by_arm[ARMS[0]]["baseline_states"])
        concatenated_baseline = np.concatenate(
            [features_by_arm[arm]["baseline_states"] for arm in ARMS], axis=0,
        )
        concatenated_evidence = np.concatenate(
            [features_by_arm[arm]["evidence_field"] for arm in ARMS], axis=0,
        )
        probabilities = hmm_posterior(
            concatenated_baseline, concatenated_evidence,
            features_by_arm[ARMS[0]]["marker_cM"],
            float(effective["hazard_per_morgan"]), float(effective["evidence_scale"]),
        )
        tune_people = tune_by_arm[0][1]
        for arm_index, arm in enumerate(ARMS):
            first, last = arm_index * arm_people, (arm_index + 1) * arm_people
            metrics[arm] = _candidate_metric(
                probabilities[first:last], tune_people, features_by_arm[arm], truth,
                candidate_id, family, root, arm,
            )
        del probabilities, concatenated_baseline, concatenated_evidence
        gc.collect()
        return metrics, tune_by_arm[0][0], tune_people

    common_tune: np.ndarray | None = None
    common_train: np.ndarray | None = None
    for arm in ARMS:
        features = features_by_arm[arm]
        training_diagnostics: dict[str, int | bool | None] = {}
        spec = TraceSpec(
            int(effective.get("hidden_dim", 32)), int(effective.get("depth", 2)),
            int(effective.get("kernel_size", 3)), float(effective.get("dropout", 0.0)),
            tuple(int(value) for value in effective.get("dilations", (1, 2))),
        )
        probabilities, tune_people = train(
            features, features, truth, family,
            float(effective.get("hazard_per_morgan", 12.0)),
            float(effective["evidence_scale"]), spec, int(effective["updates"]),
            float(effective.get("learning_rate", 0.0003)), int(effective["batch_people"]),
            int(effective["marker_shard"]), int(effective["validation_every"]),
            int(effective["early_stopping_patience"]), int(effective.get("seed", 1103)),
            checkpoint=None, tune_fraction=float(effective["tune_fraction"]),
            split_seed=int(effective["split_seed"]),
            event_radius_cm=float(effective["event_radius_cM"]),
            training_diagnostics=training_diagnostics,
        )
        train_people = np.setdiff1d(np.arange(len(features["sample_key_sha256"])),
                                    tune_people, assume_unique=True)
        if common_tune is None:
            common_tune, common_train = tune_people, train_people
        else:
            require(np.array_equal(common_tune, tune_people) and
                    np.array_equal(common_train, train_people),
                    "M37 compact TRAIN/TUNE split changed between arms")
        metrics[arm] = _candidate_metric(probabilities, tune_people, features, truth,
                                         candidate_id, family, root, arm)
        metrics[arm]["training_diagnostics"] = training_diagnostics
        del probabilities
        gc.collect()
    require(common_tune is not None and common_train is not None,
            "M37 compact candidate emitted no arm metrics")
    return metrics, common_train, common_tune


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path.name}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_family(run_id: str, family: str, root: str, manifest_path: Path,
               parent_contract_path: Path, amendment_path: Path,
               positive_control_path: Path, positive_control_receipt_path: Path,
               canonical_metrics_path: Path, canonical_receipt_path: Path,
               truth_path: Path, f0_receipt_path: Path,
               feature_paths: list[Path], receipt_paths: list[Path],
               output_dir: Path, run_overlay: Path, run_overlay_uri: str,
               container_digest: str, auth_files: list[Path]) -> dict[str, Any]:
    require(SAFE_ID.fullmatch(run_id) and family in ("hmm", "tcn") and
            root == "R0" and run_overlay_uri.strip() and "@sha256:" in container_digest,
            "M37 compact invocation identity differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, replay, candidates = _load_manifest(
        manifest_path, family, root, parent_contract_path, amendment_path,
    )
    positive_control = _json(positive_control_path)
    positive_control_receipt = _json(positive_control_receipt_path)
    positive_auth = positive_control.get("authenticated_source_sha256")
    required_positive_sources = {
        "m37_trace_compact_positive_control.py", "m37_trace_train.py",
        "m37_trace_core.py", "m33_safe_bridge_core.py",
    }
    require(positive_control.get("stage") == "M37_TRACE_COMPACT_POSITIVE_CONTROL" and
            positive_control.get("run_id") == run_id and
            positive_control.get("container_digest") == container_digest and
            positive_control.get("candidate_manifest_sha256") == sha256(manifest_path) and
            positive_control.get("parent_contract_sha256") == sha256(parent_contract_path) and
            positive_control.get("contract_amendment_sha256") == sha256(amendment_path) and
            positive_control.get("truth_or_real_features_opened") is False and
            isinstance(positive_control.get("controls", {}).get(family), dict) and
            positive_control_receipt.get("stage") == "M37_TRACE_COMPACT_POSITIVE_CONTROL" and
            positive_control_receipt.get("run_id") == run_id and
            positive_control_receipt.get("container_digest") == container_digest and
            isinstance(positive_auth, dict) and positive_auth and
            required_positive_sources <= set(positive_auth) and
            positive_control_receipt.get("authenticated_source_sha256") == positive_auth and
            positive_control_receipt.get("output_sha256") == sha256(positive_control_path),
            "M37 compact runtime positive-control evidence differs")
    family_control = positive_control["controls"][family]
    if family == "hmm":
        hmm_controls = family_control.get("candidates")
        expected_hmm_ids = {str(row["candidate_id"]) for row in candidates}
        require(family_control.get("status") ==
                "PASS_ALL_CANDIDATES_ADDITIVE_DETECTABILITY" and
                family_control.get("all_candidates_pass") is True and
                family_control.get("candidate_count") == len(candidates) and
                isinstance(hmm_controls, dict) and set(hmm_controls) == expected_hmm_ids and
                all(row.get("status") == "PASS_ADDITIVE_DETECTABILITY" and
                    row.get("pass") is True for row in hmm_controls.values()),
                "M37 HMM candidate-specific additive detectability differs")
    else:
        candidate_controls = family_control.get("candidates")
        evaluated_ids = family_control.get("evaluated_candidate_ids")
        eligible_ids = family_control.get("eligible_candidate_ids")
        expected_seeds = manifest["positive_control_status"]["tcn"]["replication_seeds"]
        expected_ladder = manifest["positive_control_status"]["tcn"]["budget_ladder_updates"]
        expected_candidate_ids = {str(row["candidate_id"]) for row in candidates}
        require(family_control.get("status") == "PASS_AT_LEAST_ONE_CANDIDATE" and
                isinstance(candidate_controls, dict) and
                isinstance(evaluated_ids, list) and isinstance(eligible_ids, list) and
                set(evaluated_ids) == set(candidate_controls) == expected_candidate_ids and
                family_control.get("candidate_count") == len(expected_candidate_ids) and
                family_control.get("budget_ladder_updates") == expected_ladder and
                0 < len(eligible_ids) <= len(evaluated_ids) and
                set(eligible_ids) <= expected_candidate_ids and
                family_control.get("replication_seeds") == expected_seeds and
                family_control.get("selection_of_best_candidate") == "FORBIDDEN" and
                family_control.get("selection_of_best_seed") == "FORBIDDEN" and
                family_control.get("scientific_closure_if_failed") == "FORBIDDEN",
                "M37 TCN candidate-specific capacity gate differs")
        for candidate_id, control in candidate_controls.items():
            seed_results = control.get("seed_results") if isinstance(control, dict) else None
            require(isinstance(seed_results, dict) and
                    control.get("seeds") == expected_seeds and
                    set(seed_results) == {str(seed) for seed in expected_seeds},
                    "M37 TCN candidate lacks all fixed-seed capacity ladders")
            for seed_result in seed_results.values():
                evaluated_updates = seed_result.get("evaluated_updates")
                first_pass_updates = seed_result.get("first_pass_updates")
                require(isinstance(evaluated_updates, list) and evaluated_updates and
                        evaluated_updates == expected_ladder[:len(evaluated_updates)] and
                        (first_pass_updates is None or
                         first_pass_updates == evaluated_updates[-1]) and
                        seed_result.get("pass") is (first_pass_updates is not None),
                        "M37 TCN candidate capacity ladder differs")
        for candidate_id in eligible_ids:
            control = candidate_controls.get(candidate_id)
            require(isinstance(control, dict) and
                    control.get("status") == "PASS_CAPACITY_2_OF_3" and
                    int(control.get("pass_count", -1)) >= 2 and
                    int(control.get("effective_updates", -1)) in expected_ladder,
                    "M37 TCN eligible candidate lacks replicated capacity evidence")
        # Only capacity-qualified candidates are allowed to touch real FIT/TUNE.
        candidates = [
            {
                **row,
                "parameters": {
                    **row.get("parameters", {}),
                    "updates": int(candidate_controls[row["candidate_id"]]["effective_updates"]),
                },
            }
            for row in candidates if row["candidate_id"] in eligible_ids
        ]
        require(candidates, "M37 TCN has no capacity-qualified real-data candidate")
        # The residual operator changed prospectively, so the non-binding TCN
        # reference uses a qualified candidate and is reused by the sweep.
        replay = {**candidates[0], "canonical_candidate_id":
                  replay["canonical_candidate_id"]}
    canonical, canonical_collection_receipt = _load_canonical_collection(
        canonical_metrics_path, canonical_receipt_path, root,
    )
    truth = load_truth(truth_path)
    features, feature_receipts, feature_evidence, baseline_provenance = _bind_features(
        feature_paths, receipt_paths, truth_path, f0_receipt_path,
    )
    require(truth.shape == features[ARMS[0]]["baseline_states"].shape[:2],
            "M37 compact FIT truth/features axes differ")

    auth_sha = {path.name: sha256(path) for path in auth_files}
    require(len(auth_sha) == len(auth_files), "M37 compact authenticated source basenames collide")
    auth_set_sha = _canonical_hash(auth_sha)
    replay_id = f"{family}_default_equivalence_replay"
    replay_effective = _effective_parameters(manifest, family, replay)
    replay_metrics, replay_train, replay_tune = _execute_candidate(
        replay_id, family, replay_effective, features, truth, root,
    )
    absolute_tolerance = float(manifest["equivalence"]["absolute_tolerance"])
    relative_tolerance = float(manifest["equivalence"]["relative_tolerance"])
    canonical_id = str(replay["canonical_candidate_id"])
    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARMS:
        require((canonical_id, arm) in canonical,
                f"M37 canonical collection lacks {canonical_id}/{arm}")
        comparisons[arm] = compare_metrics(replay_metrics[arm], canonical[(canonical_id, arm)],
                                           absolute_tolerance, relative_tolerance)
    equivalence_pass = all(row["pass"] for arm in comparisons.values() for row in arm.values())
    equivalence_policy = str(manifest["equivalence"]["policy_by_family"][family])
    equivalence_required = equivalence_policy == "REQUIRE_CANONICAL_METRIC_REPLAY"
    equivalence_path = output_dir / f"{family}.equivalence.json"
    equivalence_payload = {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_COMPACT_EQUIVALENCE",
        "status": (("PASS" if equivalence_pass else "FAIL") if equivalence_required else
                   "NEWLY_FROZEN_REFERENCE_ONLY"),
        "policy": equivalence_policy,
        "canonical_metric_match": equivalence_pass,
        "root": root,
        "run_id": run_id,
        "family": family,
        "canonical_candidate_id": canonical_id,
        "replay_candidate_id": replay_id,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "metric_paths": [".".join(path) for path in METRIC_PATHS],
        "comparisons": comparisons,
        "effective_hyperparameters": replay_effective,
        "manifest_sha256": sha256(manifest_path),
        "parent_contract_sha256": sha256(parent_contract_path),
        "contract_amendment_sha256": sha256(amendment_path),
        "positive_control_sha256": sha256(positive_control_path),
        "positive_control_receipt_sha256": sha256(positive_control_receipt_path),
        "canonical_metrics_sha256": sha256(canonical_metrics_path),
        "canonical_metrics_receipt_sha256": sha256(canonical_receipt_path),
        "truth_sha256": sha256(truth_path),
        "baseline_provenance": baseline_provenance,
        "feature_evidence": feature_evidence,
        "train_sample_axis_sha256": hashlib.sha256(
            features[ARMS[0]]["sample_key_sha256"][replay_train].tobytes()).hexdigest(),
        "tune_sample_axis_sha256": hashlib.sha256(
            features[ARMS[0]]["sample_key_sha256"][replay_tune].tobytes()).hexdigest(),
        "marker_axis_sha256": str(features[ARMS[0]]["marker_axis_sha256"].reshape(-1)[0]),
        "run_overlay": {"uri": run_overlay_uri, "sha256": sha256(run_overlay)},
        "container_digest": container_digest,
        "authenticated_source_sha256": auth_sha,
        "authenticated_source_set_sha256": auth_set_sha,
    }
    _write_json(equivalence_path, equivalence_payload)
    equivalence_receipt_path = output_dir / f"{family}.equivalence.receipt.json"
    _write_json(equivalence_receipt_path, {
        "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_EQUIVALENCE",
        "run_id": run_id, "root": root, "family": family,
        "container_digest": container_digest,
        "canonical_collection_stage": canonical_collection_receipt["stage"],
        "output_sha256": sha256(equivalence_path),
    })
    require(not equivalence_required or equivalence_pass,
            f"M37 {family} compact replay differs from canonical R0; sweep remains closed")

    metric_hashes: dict[str, str] = {}
    metric_receipt_hashes: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        effective = _effective_parameters(manifest, family, candidate)
        if effective == replay_effective:
            candidate_metrics = copy.deepcopy(replay_metrics)
            for arm in ARMS:
                candidate_metrics[arm]["candidate_id"] = candidate_id
            train_people, tune_people = replay_train, replay_tune
        else:
            candidate_metrics, train_people, tune_people = _execute_candidate(
                candidate_id, family, effective, features, truth, root,
            )
        if family == "tcn":
            _require_tcn_rd_raw_f0_identity(candidate_metrics["RD"])
        train_axis_sha = hashlib.sha256(
            features[ARMS[0]]["sample_key_sha256"][train_people].tobytes()).hexdigest()
        tune_axis_sha = hashlib.sha256(
            features[ARMS[0]]["sample_key_sha256"][tune_people].tobytes()).hexdigest()
        for arm in ARMS:
            candidate_metrics[arm]["schema_version"] = "1.0.0"
            candidate_metrics[arm]["stage"] = "M37_TRACE_SCORE"
            candidate_metrics[arm]["run_id"] = run_id
            metric_path = output_dir / f"{candidate_id}.{family}.{arm}.metrics.json"
            _write_json(metric_path, candidate_metrics[arm])
            metric_receipt_path = output_dir / f"{candidate_id}.{family}.{arm}.metrics.receipt.json"
            _write_json(metric_receipt_path, {
                "schema_version": "1.0.0",
                "stage": "M37_TRACE_SCORE",
                "candidate_id": candidate_id,
                "family": family,
                "run_id": run_id,
                "root": root,
                "arm": arm,
                "evaluation_split": "FIT_TUNE",
                "execution_mode": "IN_MEMORY_NO_PREDICTION_OR_CHECKPOINT_EXPORT",
                "effective_hyperparameters": effective,
                "training_diagnostics": candidate_metrics[arm].get(
                    "training_diagnostics",
                    "NOT_APPLICABLE_DETERMINISTIC_HMM",
                ),
                "manifest_sha256": sha256(manifest_path),
                "parent_contract_sha256": sha256(parent_contract_path),
                "contract_amendment_sha256": sha256(amendment_path),
                "positive_control_sha256": sha256(positive_control_path),
                "positive_control_receipt_sha256": sha256(positive_control_receipt_path),
                "canonical_metrics_sha256": sha256(canonical_metrics_path),
                "canonical_metrics_receipt_sha256": sha256(canonical_receipt_path),
                "equivalence_sha256": sha256(equivalence_path),
                "equivalence_receipt_sha256": sha256(equivalence_receipt_path),
                "features_sha256": feature_evidence[arm]["features_sha256"],
                "features_receipt_sha256": feature_evidence[arm]["features_receipt_sha256"],
                "features_receipt_output_sha256": feature_receipts[arm]["output_sha256"],
                "truth_sha256": sha256(truth_path),
                "baseline_provenance": baseline_provenance,
                "train_sample_axis_sha256": train_axis_sha,
                "tune_sample_axis_sha256": tune_axis_sha,
                "marker_axis_sha256": str(features[arm]["marker_axis_sha256"].reshape(-1)[0]),
                "fit_valid_sample_overlap": None,
                "persisted_predictions": 0,
                "persisted_checkpoints": 0,
                "run_overlay": {"uri": run_overlay_uri, "sha256": sha256(run_overlay)},
                "container_digest": container_digest,
                "authenticated_source_set_sha256": auth_set_sha,
                "output_sha256": sha256(metric_path),
            })
            metric_hashes[metric_path.name] = sha256(metric_path)
            metric_receipt_hashes[metric_receipt_path.name] = sha256(metric_receipt_path)

    audit_path = output_dir / f"{family}.compact_sweep.audit.json"
    audit_payload = {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_COMPACT_SWEEP",
        "status": "PASS_FIT_TUNE_ONLY",
        "root": root,
        "run_id": run_id,
        "family": family,
        "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "arms": list(ARMS),
        "candidate_count": len(candidates),
        "metric_count": len(metric_hashes),
        "equivalence_sha256": sha256(equivalence_path),
        "equivalence_receipt_sha256": sha256(equivalence_receipt_path),
        "metric_sha256": metric_hashes,
        "metric_receipt_sha256": metric_receipt_hashes,
        "manifest_sha256": sha256(manifest_path),
        "parent_contract_sha256": sha256(parent_contract_path),
        "contract_amendment_sha256": sha256(amendment_path),
        "positive_control_sha256": sha256(positive_control_path),
        "positive_control_receipt_sha256": sha256(positive_control_receipt_path),
        "truth_sha256": sha256(truth_path),
        "baseline_provenance": baseline_provenance,
        "feature_evidence": feature_evidence,
        "canonical_metrics_sha256": sha256(canonical_metrics_path),
        "canonical_metrics_receipt_sha256": sha256(canonical_receipt_path),
        "run_overlay": {"uri": run_overlay_uri, "sha256": sha256(run_overlay)},
        "container_digest": container_digest,
        "authenticated_source_sha256": auth_sha,
        "authenticated_source_set_sha256": auth_set_sha,
        "persistent_output_policy": {
            "metrics_and_receipts_only": True,
            "prediction_npz": 0,
            "checkpoints": 0,
            "valid_access": "FORBIDDEN",
        },
        "positive_control_status": family_control,
        "positive_control_all_status": positive_control["controls"],
        "capacity_qualified_candidate_ids": (
            [str(row["candidate_id"]) for row in candidates] if family == "tcn" else []
        ),
        "tcn_rd_raw_f0_identity": (True if family == "tcn" else
                                    "NOT_APPLICABLE_HMM_SMOOTHING_MODEL"),
        "positive_control_precondition": manifest["positive_control_status"].get(family),
        "decision_scope": "R0 TUNE can only support ADVANCE_EXPLORATORY or STOP_EXPLORATORY",
    }
    _write_json(audit_path, audit_payload)
    audit_receipt_path = output_dir / f"{family}.compact_sweep.audit.receipt.json"
    _write_json(audit_receipt_path, {
        "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_SWEEP",
        "run_id": run_id, "root": root, "family": family,
        "container_digest": container_digest,
        "candidate_count": len(candidates),
        "metric_count": len(metric_hashes), "output_sha256": sha256(audit_path),
    })
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("hmm", "tcn"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--contract-amendment", type=Path, required=True)
    parser.add_argument("--positive-control", type=Path, required=True)
    parser.add_argument("--positive-control-receipt", type=Path, required=True)
    parser.add_argument("--canonical-metrics", type=Path, required=True)
    parser.add_argument("--canonical-metrics-receipt", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--f0-receipt", type=Path, required=True)
    parser.add_argument("--feature", action="append", type=Path, required=True)
    parser.add_argument("--feature-receipt", action="append", type=Path, required=True)
    parser.add_argument("--run-overlay", type=Path, required=True)
    parser.add_argument("--run-overlay-uri", required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--auth-file", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_family(
        args.run_id, args.family, args.root, args.candidate_manifest,
        args.parent_contract, args.contract_amendment,
        args.positive_control, args.positive_control_receipt, args.canonical_metrics,
        args.canonical_metrics_receipt, args.truth, args.f0_receipt,
        args.feature, args.feature_receipt,
        args.output_dir, args.run_overlay, args.run_overlay_uri,
        args.container_digest, args.auth_file,
    )
    print(json.dumps({"status": result["status"], "family": args.family,
                      "candidates": result["candidate_count"],
                      "metrics": result["metric_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
