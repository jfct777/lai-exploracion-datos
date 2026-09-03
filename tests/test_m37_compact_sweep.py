from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m37_trace_collect_metrics import ARMS, collect_metrics
from m37_trace_compact_decision import apply_positive_control_gate, decide
from m37_trace_compact_sweep import (_bind_features, _candidate_metric, _effective_parameters,
                                     _load_manifest, compare_metrics, load_features,
                                     load_truth, run_family)
from m37_trace_core import TraceSpec
from m37_trace_train import train


CONTAINER = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@"
             "sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99")
F0_SHA = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_payload(arm: str) -> dict[str, np.ndarray]:
    people, markers = 10, 9
    marker_cm = np.arange(markers, dtype=np.float64) / 100.0
    baseline = np.full((people, markers, 6), .02, dtype=np.float32)
    baseline[:, :, 0] = .90
    baseline /= baseline.sum(axis=2, keepdims=True)
    evidence = np.zeros_like(baseline)
    if arm == "RE":
        evidence[:, 3:6, 3] = .2
    elif arm == "SHAM":
        evidence[:, 3:6, 5] = .2
    events = 0 if arm == "RD" else people
    event_values = np.zeros((events, 20), dtype=np.float32)
    if events and arm != "GEOMETRY":
        event_values[:, 0] = 1.0
        event_values[:, 4] = .25
    event_sample = np.arange(events, dtype=np.uint32)
    event_marker = np.full(events, 4, dtype=np.uint32)
    event_cm = np.full(events, .04, dtype=np.float64)
    return {
        "baseline_states": baseline,
        "evidence_field": evidence,
        "event_values": event_values,
        "event_context_7mer": np.zeros(events, dtype=np.uint16),
        "event_sample": event_sample,
        "event_marker_left": event_marker,
        "event_marker_right": event_marker,
        "event_delta_left_cM": np.zeros(events, dtype=np.float32),
        "event_delta_right_cM": np.zeros(events, dtype=np.float32),
        "event_cM": event_cm,
        "marker_cM": marker_cm,
        "marker_pos": np.arange(10_000, 10_000 + markers, dtype=np.int64),
        "marker_axis_sha256": np.asarray(["fixture-axis"]),
        "sample_key_sha256": np.asarray(
            [f"sample-digest-{index:02d}".encode("ascii") for index in range(people)],
            dtype="S64",
        ),
        "state_names": np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"]),
        "schedule_sample": np.arange(people, dtype=np.uint32),
        "schedule_marker": np.full(people, 4, dtype=np.uint32),
        "baseline_method": np.asarray(["fixture-F0"]),
        "baseline_source_sha256": np.asarray([F0_SHA]),
    }


def _write_inputs(root: Path) -> tuple[list[Path], list[Path], Path, Path]:
    features, receipts = [], []
    for arm in ARMS:
        feature = root / f"FIT.{arm}.trace.npz"
        np.savez(feature, **_feature_payload(arm))
        receipt = root / f"FIT.{arm}.trace.receipt.json"
        _write_json(receipt, {
            "schema_version": "1.0.0", "stage": "M37_TRACE_MATERIALIZE",
            "arm": arm, "target_ref_disjoint": True,
            "target_fold_assignment": "forbidden",
            "physical_genetic_axis_sha256": "fixture-axis",
            "inputs": {"F0_sha256": F0_SHA},
            "output_sha256": _sha(feature),
        })
        features.append(feature)
        receipts.append(receipt)
    first = _feature_payload("RE")
    truth = np.zeros((10, 9), dtype=np.uint8)
    truth[::2, 3:6] = 3
    truth_path = root / "truth.npz"
    np.savez(truth_path, state_labels=truth,
             sample_key_sha256=first["sample_key_sha256"], marker_pos=first["marker_pos"])
    f0_receipt = root / "m34_f0.receipt.json"
    _write_json(f0_receipt, {
        "schema_version": "1.0.0", "stage": "M34_PARSE_FLARE_F0",
        "decision": "PASS_F0_TRUTH_BLIND", "truth_opened": False,
        "contains_truth": False, "ancestry_order": ["AFR", "EUR", "NAM"],
        "sample_count": 10, "marker_count": 9,
        "outputs": {"fixture_f0.npz": {"sha256": F0_SHA}},
    })
    return features, receipts, truth_path, f0_receipt


def _write_contracts(root: Path) -> tuple[Path, Path, Path]:
    parent = root / "parent.json"
    _write_json(parent, {
        "schema_version": "1.0.0", "stage": "M37_TRACE_FINITE_SUCCESSIVE_HALVING",
        "rungs": [{"name": "triage", "updates": 2}],
        "training": {"batch_people": 2, "marker_shard": 9,
                     "validation_every": 1, "early_stopping_patience": 1},
        "hmm": {"hazard_per_morgan": [12.0], "evidence_lambda": [1.0]},
        "tcn": {"hidden_dim": [32], "depth": [2], "kernel_size": [3],
                "dropout": [0.0], "learning_rate": [0.0003],
                "seeds": [1103], "dilations": [[1, 2]]},
    })
    amendment = root / "amendment.json"
    _write_json(amendment, {
        "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_SWEEP_AMENDMENT",
        "parent_contract_sha256": _sha(parent),
        "candidate_count_by_family": {"hmm": 1, "tcn": 1},
        "hmm": {"hazard_per_morgan": [12.0], "evidence_lambda": [1.0]},
        "tcn": {"event_radius_cM": [0.02], "evidence_lambda": [1.0],
                "candidate_ids": ["tcn_fixture"],
                "execution": {"validation_every": 1,
                              "early_stopping_patience": 1,
                              "budget_ladder_updates": [2, 4, 8, 16]}},
        "capacity_control": {
            "thresholds": {
                "hmm_additive_maximum_posterior_change": 1e-4,
                "additive_balanced_accuracy": .8,
                "xor_balanced_accuracy": .75,
                "xor_one_bit_ablation_maximum_distance_from_chance": .1,
                "zero_revival_mean_probability": .5,
                "zero_revival_log_loss_gain": .5,
            },
            "screen_seed": 1103,
            "screen_candidate_count": 1,
            "budget_ladder_updates": [2, 4, 8, 16],
            "rung_execution": "DETERMINISTIC_RESTART_SAME_SEED_EACH_RUNG",
            "candidate_evaluation":
                "ALL_DECLARED_CANDIDATES_ACROSS_ALL_FIXED_SEEDS_NO_RANKING",
            "replication_seeds": [1103, 2207, 3301],
            "valid_access": "FORBIDDEN",
        },
    })
    manifest = root / "manifest.json"
    _write_json(manifest, {
        "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_SWEEP_PRE",
        "status": "PREREGISTERED_FIT_TUNE_ONLY",
        "contract_binding": {"parent_sha256": _sha(parent),
                             "amendment_sha256": _sha(amendment)},
        "scope": {"root": "R0", "evaluation_split": "FIT_TUNE",
                  "valid_access": "FORBIDDEN"},
        "arms": list(ARMS),
        "execution": {"updates": 2, "batch_people": 2, "marker_shard": 9,
                      "validation_every": 1, "early_stopping_patience": 1,
                      "tune_fraction": .2, "split_seed": 3401103,
                      "event_radius_cM": .02},
        "families": {
            "hmm": {"hazard_per_morgan": 12.0, "evidence_scale": 1.0},
            "tcn": {"evidence_scale": 1.0, "hidden_dim": 32, "depth": 2,
                    "kernel_size": 3, "dropout": 0.0, "dilations": [1, 2],
                    "seed": 1103, "learning_rate": .0003},
        },
        "equivalence": {"absolute_tolerance": 1e-6, "relative_tolerance": 1e-6,
                        "policy_by_family": {
                            "hmm": "REQUIRE_CANONICAL_METRIC_REPLAY",
                            "tcn": "REQUIRE_CANONICAL_METRIC_REPLAY",
                        },
                        "replay": {
                            "hmm": {"family": "hmm", "canonical_candidate_id": "hmm_r0",
                                    "parameters": {}},
                            "tcn": {"family": "tcn", "canonical_candidate_id": "tcn_r0",
                                    "parameters": {}},
                        }},
        "positive_control_status": {"hmm": {"status": "NA"},
                                    "tcn": {
                                        "status_across_budget_ladder": "FIXTURE_ONLY",
                                        "screen_seed": 1103,
                                        "candidate_count": 1,
                                        "candidate_evaluation":
                                            "ALL_DECLARED_CANDIDATES_ACROSS_ALL_FIXED_SEEDS_NO_RANKING",
                                        "budget_ladder_updates": [2, 4, 8, 16],
                                        "rung_execution":
                                            "DETERMINISTIC_RESTART_SAME_SEED_EACH_RUNG",
                                        "effective_budget_rule":
                                            "SECOND_SMALLEST_FIRST_PASS_RUNG_SHARED_BY_ALL_FIVE_ARMS",
                                        "replication_seeds": [1103, 2207, 3301],
                                        "required_seed_passes": 2,
                                        "required_controls": [
                                            "additive", "xor_interaction",
                                            "xor_one_bit_ablation", "zero_revival",
                                        ],
                                    }},
        "candidates": [
            {"candidate_id": "hmm_fixture", "family": "hmm", "parameters": {}},
            {"candidate_id": "tcn_fixture", "family": "tcn", "parameters": {}},
        ],
    })
    return parent, amendment, manifest


def _write_positive_control(root: Path, run_id: str, manifest: Path,
                            parent: Path, amendment: Path) -> tuple[Path, Path]:
    """Write a lightweight, fully bound fixture; production computes these controls."""
    artifact = root / "m37.compact_positive_control.json"
    auth = {
        name: _sha(ROOT / "bin" / name) for name in (
            "m37_trace_compact_positive_control.py", "m37_trace_train.py",
            "m37_trace_core.py", "m33_safe_bridge_core.py",
        )
    }
    _write_json(artifact, {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_COMPACT_POSITIVE_CONTROL",
        "run_id": run_id,
        "budget": {"updates": 2},
        "container_digest": CONTAINER,
        "candidate_manifest_sha256": _sha(manifest),
        "parent_contract_sha256": _sha(parent),
        "contract_amendment_sha256": _sha(amendment),
        "authenticated_source_sha256": auth,
        "truth_or_real_features_opened": False,
        "controls": {
            "hmm": {
                "status": "PASS_ALL_CANDIDATES_ADDITIVE_DETECTABILITY",
                "candidate_count": 1,
                "all_candidates_pass": True,
                "candidates": {
                    "hmm_fixture": {
                        "status": "PASS_ADDITIVE_DETECTABILITY", "pass": True,
                    },
                },
            },
            "tcn": {
                "status": "PASS_AT_LEAST_ONE_CANDIDATE",
                "candidate_count": 1,
                "budget_ladder_updates": [2, 4, 8, 16],
                "evaluated_candidate_ids": ["tcn_fixture"],
                "eligible_candidate_ids": ["tcn_fixture"],
                "replication_seeds": [1103, 2207, 3301],
                "selection_of_best_candidate": "FORBIDDEN",
                "selection_of_best_seed": "FORBIDDEN",
                "candidates": {
                    "tcn_fixture": {
                        "status": "PASS_CAPACITY_2_OF_3", "pass_count": 2,
                        "effective_updates": 2,
                        "seeds": [1103, 2207, 3301],
                        "seed_results": {
                            "1103": {"pass": True, "first_pass_updates": 2,
                                     "evaluated_updates": [2]},
                            "2207": {"pass": True, "first_pass_updates": 2,
                                     "evaluated_updates": [2]},
                            "3301": {"pass": False, "first_pass_updates": None,
                                     "evaluated_updates": [2, 4, 8, 16]},
                        },
                    },
                },
                "scientific_closure_if_failed": "FORBIDDEN",
            },
        },
    })
    receipt = root / "m37.compact_positive_control.receipt.json"
    _write_json(receipt, {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_COMPACT_POSITIVE_CONTROL",
        "run_id": run_id,
        "container_digest": CONTAINER,
        "authenticated_source_sha256": auth,
        "output_sha256": _sha(artifact),
    })
    return artifact, receipt


def _canonical_metrics(root: Path, manifest_path: Path, truth_path: Path,
                       feature_paths: list[Path]) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth = load_truth(truth_path)
    features = {arm: load_features(path) for arm, path in zip(ARMS, feature_paths)}
    rows = []
    for family, canonical_id in (("hmm", "hmm_r0"), ("tcn", "tcn_r0")):
        effective = _effective_parameters(manifest, family,
                                          manifest["equivalence"]["replay"][family])
        for arm in ARMS:
            item = features[arm]
            probabilities, tune = train(
                item, item, truth, family, float(effective.get("hazard_per_morgan", 12.0)),
                float(effective["evidence_scale"]), TraceSpec(32, 2, 3, 0.0, (1, 2)),
                int(effective["updates"]), float(effective.get("learning_rate", .0003)),
                int(effective["batch_people"]), int(effective["marker_shard"]),
                int(effective["validation_every"]), int(effective["early_stopping_patience"]),
                int(effective.get("seed", 1103)), tune_fraction=float(effective["tune_fraction"]),
                split_seed=int(effective["split_seed"]),
                event_radius_cm=float(effective["event_radius_cM"]),
            )
            metric = _candidate_metric(probabilities, tune, item, truth,
                                       canonical_id, family, "R0", arm)
            rows.append({"candidate_id": canonical_id, "family": family,
                         "root": "R0", "arm": arm, "metrics": metric})
    collection = root / "canonical.json"
    _write_json(collection, {"schema_version": "1.0.0",
                             "stage": "M37_TRACE_COLLECT_METRICS", "root": "R0",
                             "evaluation_split": "FIT_TUNE", "rows": rows,
                             "input_evidence": {}})
    receipt = root / "canonical.receipt.json"
    _write_json(receipt, {"schema_version": "1.0.0",
                          "stage": "M37_TRACE_COLLECT_METRICS", "root": "R0",
                          "evaluation_split": "FIT_TUNE", "row_count": len(rows),
                          "output_sha256": _sha(collection)})
    return collection, receipt


def test_production_candidate_manifest_is_bound_and_balanced() -> None:
    manifest = ROOT / "conf/m37_trace_compact_candidates.json"
    parent = ROOT / "conf/m37_trace_sweep_contract.json"
    amendment = ROOT / "conf/m37_trace_compact_sweep_amendment.json"
    manifest_payload, _, hmm = _load_manifest(manifest, "hmm", "R0", parent, amendment)
    _, _, tcn = _load_manifest(manifest, "tcn", "R0", parent, amendment)
    assert len(hmm) == 12 and len(tcn) == 6
    hmm_effective = [_effective_parameters(manifest_payload, "hmm", row) for row in hmm]
    tcn_effective = [_effective_parameters(manifest_payload, "tcn", row) for row in tcn]
    assert {(row["hazard_per_morgan"], row["evidence_scale"])
            for row in hmm_effective} == {
                (hazard, scale) for hazard in (6.0, 12.0, 24.0)
                for scale in (0.25, 0.5, 1.0, 2.0)
            }
    assert Counter(row["hidden_dim"] for row in tcn_effective) == {32: 2, 64: 2, 96: 2}
    assert Counter(row["depth"] for row in tcn_effective) == {2: 2, 3: 2, 4: 2}
    assert Counter(row["kernel_size"] for row in tcn_effective) == {3: 3, 5: 3}
    assert Counter(row["dropout"] for row in tcn_effective) == {0.0: 2, 0.1: 2, 0.2: 2}
    assert Counter(row["learning_rate"] for row in tcn_effective) == {
        0.0001: 2, 0.0003: 2, 0.001: 2,
    }
    assert Counter(row["event_radius_cM"] for row in tcn_effective) == {
        0.05: 1, 0.1: 1, 0.2: 2, 0.5: 2,
    }
    assert all(row["updates"] == 200 and row["seed"] == 1103 and
               row["dilations"] == [1, 2, 4, 8][:row["depth"]]
               for row in tcn_effective)


def test_compact_runner_replays_defaults_and_emits_only_metric_evidence() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        features, receipts, truth, f0_receipt = _write_inputs(root)
        parent, amendment, manifest = _write_contracts(root)
        positive, positive_receipt = _write_positive_control(
            root, "fixture-run", manifest, parent, amendment,
        )
        canonical, canonical_receipt = _canonical_metrics(root, manifest, truth, features)
        overlay = root / "overlay.config"
        overlay.write_text("params.m37_root = 'R0'\n", encoding="utf-8")
        auth = [ROOT / "bin/m37_trace_compact_sweep.py",
                ROOT / "bin/m37_trace_train.py", ROOT / "bin/m37_trace_score.py"]
        metric_paths, metric_receipts, audits = [], [], []
        for family in ("hmm", "tcn"):
            output = root / family
            result = run_family(
                "fixture-run", family, "R0", manifest, parent, amendment,
                positive, positive_receipt, canonical, canonical_receipt, truth,
                f0_receipt, features, receipts, output, overlay,
                "repo://fixture/overlay.config", CONTAINER, auth,
            )
            assert result["status"] == "PASS_FIT_TUNE_ONLY"
            assert result["metric_count"] == 5
            assert not list(output.glob("*.npz")) and not list(output.glob("*.pt"))
            observed_metrics = sorted(output.glob("*.metrics.json"))
            observed_receipts = sorted(output.glob("*.metrics.receipt.json"))
            assert len(observed_metrics) == len(observed_receipts) == 5
            receipt_payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in observed_receipts
            ]
            if family == "tcn":
                # The capacity gate fixes one candidate-specific budget.  RE
                # and every paired control must then share both that budget
                # and the early-stopping schedule.
                observed_training_schedules = {
                    (
                        row["effective_hyperparameters"]["updates"],
                        row["effective_hyperparameters"]["validation_every"],
                        row["effective_hyperparameters"]["early_stopping_patience"],
                    )
                    for row in receipt_payloads
                }
                assert observed_training_schedules == {(2, 1, 1)}
                for row in receipt_payloads:
                    diagnostic = row["training_diagnostics"]
                    assert diagnostic["requested_updates"] == 2
                    assert 1 <= diagnostic["completed_updates"] <= 2
                    assert diagnostic["best_checkpoint_update"] in {1, 2}
                    assert diagnostic["validation_every"] == 1
                    assert diagnostic["early_stopping_patience"] == 1
                    assert diagnostic["early_stopping_enabled"] is True
                    assert diagnostic["restore_best_checkpoint"] is True
            first = json.loads(observed_metrics[0].read_text(encoding="utf-8"))
            assert first["schema_version"] == "1.0.0"
            assert first["stage"] == "M37_TRACE_SCORE"
            assert first["run_id"] == "fixture-run" and first["per_individual"]
            assert first["baseline_metadata"]["method"] == "FLARE"
            assert first["baseline_metadata"]["source_sha256"] == F0_SHA
            assert first["baseline_metadata"]["upstream_stage"] == "M34_PARSE_FLARE_F0"
            assert set(first["per_individual"][0]["boundary_counts"]) == {
                "0.05", "0.1", "0.2", "0.5"
            }
            metric_paths.extend(observed_metrics)
            metric_receipts.extend(observed_receipts)
            first_receipt = receipt_payloads[0]
            if family == "tcn":
                assert first["training_diagnostics"] == first_receipt[
                    "training_diagnostics"
                ]
            assert first_receipt["output_sha256"] == _sha(observed_metrics[0])
            assert first_receipt["manifest_sha256"] == _sha(manifest)
            assert first_receipt["contract_amendment_sha256"] == _sha(amendment)
            assert first_receipt["baseline_provenance"]["upstream_receipt_sha256"] == _sha(f0_receipt)
            audits.append(json.loads((output / f"{family}.compact_sweep.audit.json").read_text()))
        rows, _ = collect_metrics(metric_paths, metric_receipts, "R0", "FIT_TUNE")
        decisions, criteria = decide(rows, "R0")
        decisions, criteria = apply_positive_control_gate(
            decisions, criteria,
            audits[0]["positive_control_all_status"],
        )
        assert len(decisions) == 2
        assert {row["status"] for row in decisions} <= {"ADVANCE_EXPLORATORY", "STOP_EXPLORATORY"}
        assert all(not row["scientific_closure"] for row in decisions)
        assert criteria["pareto_only_promotion"] == "FORBIDDEN"
        assert {row["family"] for row in audits} == {"hmm", "tcn"}
        tcn_decision = next(row for row in decisions if row["family"] == "tcn")
        assert tcn_decision["capacity_control_pass"]
        assert tcn_decision["capacity_control"]["pass_count"] == 2


def test_failed_tcn_capacity_does_not_block_hmm_exploratory_decisions() -> None:
    baseline = {"f1_boundary": {"0.2": .1}, "log_loss": .4,
                "brier": .2, "macro_ancestry_dose_mae": .2,
                "ancestry_dose_mae": {"NAM": .2},
                "false_transitions_per_morgan": 1.0}
    metric = {**baseline, "baseline": baseline, "evaluation_split": "FIT_TUNE",
              "root": "R0", "candidate_id": "hmm_fixture", "arm": "RE",
              "run_id": "fixture"}
    arms = []
    for arm in ARMS:
        row_metric = json.loads(json.dumps(metric))
        row_metric["arm"] = arm
        row_metric["per_individual"] = [{
            "sample_key_sha256": "a",
            "boundary_counts": {
                value: {"TP": 0, "FP": 0, "FN": 0}
                for value in ("0.05", "0.1", "0.2", "0.5")
            },
        }]
        arms.append({"candidate_id": "hmm_fixture", "family": "hmm",
                     "root": "R0", "arm": arm, "metrics": row_metric})
    decisions, criteria = decide(arms, "R0")
    decisions, criteria = apply_positive_control_gate(decisions, criteria, {
        "hmm": {
            "status": "PASS_ALL_CANDIDATES_ADDITIVE_DETECTABILITY",
            "all_candidates_pass": True,
            "candidates": {"hmm_fixture": {
                "status": "PASS_ADDITIVE_DETECTABILITY", "pass": True,
            }},
        },
        "tcn": {
            "status": "FAIL_NO_CAPABLE_CANDIDATE", "candidates": {},
            "eligible_candidate_ids": [],
        },
    })
    assert len(decisions) == 1 and decisions[0]["family"] == "hmm"
    assert criteria["positive_control_precondition"]["tcn_real_data_action"] == (
        "TCN_REAL_DATA_NOT_RUN"
    )


def test_compact_manifest_rejects_any_valid_scope() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        parent, amendment, manifest = _write_contracts(root)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["scope"]["evaluation_split"] = "VALID"
        payload["scope"]["valid_access"] = "ALLOWED"
        _write_json(manifest, payload)
        try:
            _load_manifest(manifest, "tcn", "R0", parent, amendment)
        except ValueError as error:
            assert "only R0 FIT/TUNE" in str(error)
        else:
            raise AssertionError("a VALID-scoped compact manifest must be rejected")


def test_compact_features_require_the_exact_truth_blind_flare_receipt() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        features, receipts, truth, f0_receipt = _write_inputs(root)
        payload = json.loads(f0_receipt.read_text(encoding="utf-8"))
        payload["outputs"]["fixture_f0.npz"]["sha256"] = "b" * 64
        _write_json(f0_receipt, payload)
        try:
            _bind_features(features, receipts, truth, f0_receipt)
        except ValueError as error:
            assert "truth-blind FLARE receipt" in str(error)
        else:
            raise AssertionError("an unbound F0 receipt must be rejected")


def test_equivalence_detects_numeric_drift_beyond_tolerance() -> None:
    observed = {"log_loss": 1.0, "brier": 1.0, "macro_ancestry_dose_mae": 1.0,
                "calibration_ece_15": 1.0, "false_transitions_per_morgan": 1.0,
                "mean_boundary_error_cM": None,
                "ancestry_dose_mae": {"AFR": 1.0, "EUR": 1.0, "NAM": 1.0},
                "f1_boundary": {key: 1.0 for key in ("0.05", "0.1", "0.2", "0.5")}}
    observed["baseline"] = json.loads(json.dumps(observed))
    expected = json.loads(json.dumps(observed))
    expected["log_loss"] = 1.01
    comparison = compare_metrics(observed, expected, 1e-6, 1e-6)
    assert not comparison["log_loss"]["pass"]
