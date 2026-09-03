from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m37_trace_collect_metrics import ARMS, collect_metrics
from m37_trace_compact_decision import decide
from m37_trace_compact_sweep import (_candidate_metric, _effective_parameters,
                                     _load_manifest, compare_metrics, load_features,
                                     load_truth, run_family)
from m37_trace_core import TraceSpec
from m37_trace_train import train


CONTAINER = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@"
             "sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99")


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
        "baseline_source_sha256": np.asarray(["fixture-source"]),
    }


def _write_inputs(root: Path) -> tuple[list[Path], list[Path], Path]:
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
    return features, receipts, truth_path


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
                "execution": {"updates": 2, "validation_every": 1,
                              "early_stopping_patience": 1}},
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
                        "replay": {
                            "hmm": {"family": "hmm", "canonical_candidate_id": "hmm_r0",
                                    "parameters": {}},
                            "tcn": {"family": "tcn", "canonical_candidate_id": "tcn_r0",
                                    "parameters": {}},
                        }},
        "positive_control_status": {"hmm": {"status": "NA"},
                                    "tcn": {"status_at_2_updates": "FIXTURE_ONLY",
                                            "anchor_candidate_id": "tcn_fixture"}},
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
        "controls": {
            "hmm": {"status": "PASS_ADDITIVE_DETECTABILITY"},
            "tcn": {
                "anchor_candidate_id": "tcn_fixture",
                "updates": 2,
                "architecture": {
                    "hidden_dim": 32, "depth": 2, "kernel_size": 3,
                    "dropout": 0.0, "dilations": [1, 2],
                    "learning_rate": .0003, "seed": 1103,
                    "event_radius_cM": .02, "evidence_scale": 1.0,
                    "validation_every": 1, "early_stopping_patience": 1,
                },
                "additive": {"status": "PASS"},
                "xor_interaction": {"status": "BUDGET_INSUFFICIENT_FOR_INTERACTION"},
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
    _, _, hmm = _load_manifest(manifest, "hmm", "R0", parent, amendment)
    _, _, tcn = _load_manifest(manifest, "tcn", "R0", parent, amendment)
    assert len(hmm) == 20 and len(tcn) == 3


def test_compact_runner_replays_defaults_and_emits_only_metric_evidence() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        features, receipts, truth = _write_inputs(root)
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
                features, receipts, output, overlay,
                "repo://fixture/overlay.config", CONTAINER, auth,
            )
            assert result["status"] == "PASS_FIT_TUNE_ONLY"
            assert result["metric_count"] == 5
            assert not list(output.glob("*.npz")) and not list(output.glob("*.pt"))
            observed_metrics = sorted(output.glob("*.metrics.json"))
            observed_receipts = sorted(output.glob("*.metrics.receipt.json"))
            assert len(observed_metrics) == len(observed_receipts) == 5
            first = json.loads(observed_metrics[0].read_text(encoding="utf-8"))
            assert first["run_id"] == "fixture-run" and first["per_individual"]
            assert set(first["per_individual"][0]["boundary_counts"]) == {
                "0.05", "0.1", "0.2", "0.5"
            }
            metric_paths.extend(observed_metrics)
            metric_receipts.extend(observed_receipts)
            audits.append(json.loads((output / f"{family}.compact_sweep.audit.json").read_text()))
        rows, _ = collect_metrics(metric_paths, metric_receipts, "R0", "FIT_TUNE")
        decisions, criteria = decide(rows, "R0")
        assert len(decisions) == 2
        assert {row["status"] for row in decisions} <= {"ADVANCE_EXPLORATORY", "STOP_EXPLORATORY"}
        assert all(not row["scientific_closure"] for row in decisions)
        assert criteria["pareto_only_promotion"] == "FORBIDDEN"
        assert {row["family"] for row in audits} == {"hmm", "tcn"}


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
