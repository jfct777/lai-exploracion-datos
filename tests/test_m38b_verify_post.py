from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m38b_verify_post as subject  # noqa: E402


def metric_intervals(lower: float = -0.1, upper: float = 0.1) -> dict[str, list[float]]:
    return {
        metric: [lower, upper]
        for metric in subject.ERROR_VETO_METRICS + ("boundary_f1_0_2cm",)
    }


def contrast_row(value: float = -0.1, upper: float = -0.01) -> dict:
    n_eff = {
        metric: {"0": 32, "1": 32, "2": 32}
        for metric in subject.ERROR_VETO_METRICS + ("boundary_f1_0_2cm",)
    }
    return {
        "fold_mean_deltas": {"0": value, "1": value, "2": value},
        "one_sided_upper_97_5_two_family": upper,
        "metric_deltas_left_minus_right": {"log_loss_uniform": value},
        "metric_fold_mean_deltas": {
            "log_loss_uniform": {"0": value, "1": value, "2": value},
        },
        "metric_delta_percentile_ci95": metric_intervals(),
        "metric_fold_n_eff": n_eff,
        "negative_direction_folds": 3 if value < 0 else 0,
        "direction_3_of_3": value < 0,
        "candidate_contrast_gate": value < 0 and upper < 0,
    }


def family_report() -> dict:
    report = {
        "contrasts": {
            "full-minus": {},
            "RE-RD": contrast_row(),
            "RE-SHAM": contrast_row(),
            "RE-full": contrast_row(),
        },
        "candidate_incremental_gate": {"pass": True},
        "secondary_gates": {
            "weighted_uniform_no_sign_reversal": {"pass": True},
            "no_statistically_clear_harm": {"pass": True},
            "no_statistically_clear_harm_vs_full": {"pass": True},
            "deploy_improvement_over_full_flare": {"pass": True},
        },
    }
    return report


def positive_report(value: float = -0.1, upper: float = -0.01) -> dict:
    contrasts = {}
    for logical_id in subject.EXPECTED_POSITIVE_IDS[1:]:
        passed = value < 0 and upper < 0
        contrasts[logical_id] = {
            "fold_mean_deltas": {"0": value, "1": value, "2": value},
            "one_sided_upper_98_75": upper,
            "favorable_folds": 3 if value < 0 else 0,
            "bonferroni_four_delta_gate": passed,
        }
    return {
        "logical_ids": list(subject.EXPECTED_POSITIVE_IDS),
        "contrasts": contrasts,
        "capacity_gate": {"pass": value < 0 and upper < 0},
    }


def feature_payload() -> dict[str, np.ndarray]:
    return {
        "sample_key_sha256": np.asarray(["p0", "p1"]),
        "marker_pos": np.asarray([100, 200, 300], dtype=np.int64),
        "marker_cM": np.asarray([0.0, 0.07, 0.2], dtype=np.float64),
        "marker_axis_sha256": np.asarray(["a" * 64]),
        "state_names": np.asarray(subject.STATE_NAMES),
        "event_sample": np.asarray([0, 1], dtype=np.int64),
        "event_locus": np.asarray([0, 1], dtype=np.int64),
        "event_cM": np.asarray([0.05, 0.15], dtype=np.float64),
        "event_marker_left": np.asarray([0, 1], dtype=np.int64),
        "event_marker_right": np.asarray([1, 2], dtype=np.int64),
        "event_delta_left_cM": np.asarray([0.05, 0.08], dtype=np.float64),
        "event_delta_right_cM": np.asarray([0.02, 0.05], dtype=np.float64),
        "schedule_sample": np.asarray([0, 1], dtype=np.int64),
        "schedule_marker": np.asarray([1, 2], dtype=np.int64),
        "event_target_callable": np.asarray([True, True]),
        "event_reference_callable": np.asarray([True, False]),
        "evidence_field": np.zeros((2, 3, 6), dtype=np.float32),
        "event_values": np.zeros((2, 23), dtype=np.float32),
        "event_context_7mer": np.zeros(2, dtype=np.uint16),
        "event_genotype": np.zeros(2, dtype=np.uint8),
        "event_pooled_loglik": np.zeros((2, 6), dtype=np.float32),
        "event_loglik": np.zeros((2, 6), dtype=np.float32),
        "event_uncertainty": np.zeros((2, 6), dtype=np.float32),
        "event_support": np.zeros((2, 3), dtype=np.float32),
        "event_carrier_support": np.zeros(2, dtype=np.float32),
        "event_origin_support": np.zeros(2, dtype=np.float32),
        "event_counts": np.zeros((2, 2), dtype=np.int64),
        "context_7mer_available": np.zeros(1, dtype=np.uint8),
        "carrier_support_available": np.zeros(1, dtype=np.uint8),
        "origin_support_available": np.zeros(1, dtype=np.uint8),
    }


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def make_partition_binding(
    run_dir: Path, identity: str, fold: int, inner_seed: int,
) -> dict:
    is_positive = identity in subject.EXPECTED_POSITIVE_IDS
    arm = "POSITIVE" if is_positive else identity
    feature_stem = run_dir / f"partitions/features/m38b.{identity}.fold{fold}"
    fit_features = Path(f"{feature_stem}.fit.features.npz")
    score_features = Path(f"{feature_stem}.score.features.npz")
    feature_receipt_path = Path(f"{feature_stem}.features.receipt.json")
    write_npz(fit_features, {"rows": np.arange(64, dtype=np.int64)})
    write_npz(score_features, {"rows": np.arange(32, dtype=np.int64)})
    write_json(feature_receipt_path, {
        "stage": "M38B_PARTITION_FEATURES",
        "status": "PASS_TRUTH_BLIND_FEATURE_PARTITION",
        "fold": fold,
        "arm": arm,
        "source_arm": arm,
        "source_stage": ("M38B_POSITIVE_CONTROL_MATERIALIZE" if is_positive
                         else "M37_TRACE_MATERIALIZE"),
        "diagnostic_only": is_positive,
        "positive_control_delta": (subject.DELTA_IDS[identity] if is_positive else None),
        "inner_split_seed": inner_seed,
        "fit_output_sha256": subject.sha256_file(fit_features),
        "score_output_sha256": subject.sha256_file(score_features),
        "fit_people": 64,
        "score_people": 32,
        "truth_read": False,
    })

    truth_stem = run_dir / f"partitions/truth/m38b.fold{fold}"
    fit_truth = Path(f"{truth_stem}.fit.truth.npz")
    truth_receipt_path = Path(f"{truth_stem}.truth.receipt.json")
    write_npz(fit_truth, {"rows": np.arange(64, dtype=np.int64)})
    write_json(truth_receipt_path, {
        "stage": "M38B_PARTITION_TRUTH",
        "status": "PASS_NON_SELECTING_TRUTH_PARTITION",
        "fold": fold,
        "inner_split_seed": inner_seed,
        "fit_output_sha256": subject.sha256_file(fit_truth),
        "fit_people": 64,
        "score_people": 32,
        "model_selection_performed": False,
    })
    return {
        "diagnostic_only": is_positive,
        "inner_split_seed": inner_seed,
        "train_people": 48,
        "select_people": 16,
        "score_people": 32,
        "score_truth_input": None,
        "fit_features_sha256": subject.sha256_file(fit_features),
        "score_features_sha256": subject.sha256_file(score_features),
        "feature_receipt_sha256": subject.sha256_file(feature_receipt_path),
        "fit_truth_sha256": subject.sha256_file(fit_truth),
        "truth_receipt_sha256": subject.sha256_file(truth_receipt_path),
    }


def make_positive_materializations(run_dir: Path) -> None:
    real_path = run_dir / "features/m38b.RE.trace.npz"
    real_receipt = run_dir / "features/m38b.RE.trace.receipt.json"
    payload = feature_payload()
    write_npz(real_path, payload)
    write_json(real_receipt, {
        "arm": "RE", "output_sha256": subject.sha256_file(real_path),
    })
    event_hash = subject._array_bundle_sha256(payload, subject.EVENT_IDENTITY_MEMBERS)
    mask_hash = subject._array_bundle_sha256(payload, subject.EVENT_MASK_MEMBERS)
    axis_hash = subject._array_bundle_sha256(payload, subject.AXIS_MEMBERS)
    for fold in range(3):
        for logical_id, delta in subject.DELTA_IDS.items():
            stem = run_dir / f"controls/positive/features/m38b.{logical_id}.fold{fold}"
            path = Path(f"{stem}.npz")
            receipt = Path(f"{stem}.receipt.json")
            write_npz(path, payload)
            write_json(receipt, {
                "stage": "M38B_POSITIVE_CONTROL_MATERIALIZE",
                "status": "PASS_PRODUCTION_MATCHED_DIAGNOSTIC_CONTROL",
                "arm": "POSITIVE", "diagnostic_only": True,
                "fold": fold, "delta": delta,
                "axis_sha256": axis_hash,
                "real_event_identity_sha256": event_hash,
                "real_event_masks_sha256": mask_hash,
                "inputs": {
                    "real_features_sha256": subject.sha256_file(real_path),
                    "real_receipt_sha256": subject.sha256_file(real_receipt),
                },
                "output_sha256": subject.sha256_file(path),
            })


def make_analytic_oof(run_dir: Path) -> dict[str, str]:
    provenance = {name: (str(index + 1) * 64)[:64]
                  for index, name in enumerate(subject.PROVENANCE_FIELDS)}
    people = np.asarray([f"p{index:03d}" for index in range(96)], dtype="S64")
    marker_pos = np.asarray([100, 200, 400, 800], dtype=np.int64)
    marker_cm = np.asarray([0.0, 0.03, 0.11, 0.25], dtype=np.float64)
    marker_hash = np.asarray(["a" * 64])
    oof_probability = np.empty((96, 4, 6), dtype=np.float32)
    fold_ids = np.empty(96, dtype=np.uint8)
    sources = []
    for fold in range(3):
        partition_binding = make_partition_binding(run_dir, "RE", fold, 9000 + fold)
        selected = np.arange(fold * 32, (fold + 1) * 32)
        probability = np.full((32, 4, 6), 0.1, dtype=np.float32)
        probability[:, :, fold] = 0.5
        stem = run_dir / f"predictions/folds/m38b.analytic.RE.fold{fold}.seed1103"
        prediction = Path(f"{stem}.prediction.npz")
        prediction_receipt = Path(f"{stem}.prediction.receipt.json")
        write_npz(prediction, {
            "probabilities": probability,
            "sample_key_sha256": people[selected],
            "marker_pos": marker_pos,
            "marker_cM": marker_cm,
            "marker_axis_sha256": marker_hash,
            "fold": np.asarray([fold], dtype=np.uint8),
            "family": np.asarray(["analytic"]),
            "arm": np.asarray(["RE"]),
            "seed": np.asarray([1103], dtype=np.int64),
        })
        write_json(prediction_receipt, {
            "stage": "M38B_TRAIN_AND_PREDICT_OOF",
            "status": "PASS_SCORE_TRUTH_INACCESSIBLE",
            "fold": fold, "family": "analytic", "arm": "RE", "seed": 1103,
            "positive_control_delta": None, "diagnostic_only": False,
            "real_event_identity_sha256": None, "real_event_masks_sha256": None,
            "checkpoint_sha256": None,
            "model_contract_receipt_sha256": provenance[
                "model_contract_receipt_sha256"
            ],
            "base_contract_sha256": provenance["base_contract_sha256"],
            "amendment_sha256": provenance["amendment_sha256"],
            "amendment_2_sha256": provenance["amendment_2_sha256"],
            "output_sha256": subject.sha256_file(prediction),
            **partition_binding,
        })
        sources.append({
            "fold": fold, "seed": 1103,
            "prediction_sha256": subject.sha256_file(prediction),
            "prediction_receipt_sha256": subject.sha256_file(prediction_receipt),
        })
        oof_probability[selected] = probability
        fold_ids[selected] = fold
    oof = run_dir / "predictions/oof/m38b.analytic.RE.oof.npz"
    oof_receipt = run_dir / "predictions/oof/m38b.analytic.RE.oof.receipt.json"
    write_npz(oof, {
        "probabilities": oof_probability,
        "sample_key_sha256": people,
        "marker_pos": marker_pos,
        "marker_cM": marker_cm,
        "marker_axis_sha256": marker_hash,
        "fold_ids": fold_ids,
        "family": np.asarray(["analytic"]),
        "arm": np.asarray(["RE"]),
        "state_names": np.asarray(subject.STATE_NAMES),
        "seed_values": np.asarray([1103], dtype=np.int64),
    })
    write_json(oof_receipt, {
        "stage": "M38B_COLLECT_TRUTH_BLIND_OOF",
        "status": "PASS_EXACT_ONE_OOF_PREDICTION_PER_PERSON",
        "family": "analytic", "arm": "RE", "positive_control_delta": None,
        "diagnostic_only": False,
        "real_event_identity_sha256": None, "real_event_masks_sha256": None,
        "people": 96, "folds": 3, "score_people_per_fold": 32,
        "seeds": [1103], "person_coverage_min": 1, "person_coverage_max": 1,
        "truth_input": None, "truth_read": False,
        "selector_or_checkpoint_access": False,
        "state_names": list(subject.STATE_NAMES), "sources": sources,
        "output_sha256": subject.sha256_file(oof), **provenance,
    })
    return provenance


class M38BPostVerifierTests(unittest.TestCase):
    def test_publication_inventory_contains_exactly_356_expected_files(self) -> None:
        observed = subject.expected_relative_paths()
        self.assertEqual(len(observed), 356)
        self.assertIn("prelaunch/m38b.model_contract.receipt.json", observed)
        self.assertIn("prelaunch/run_provenance.receipt.json", observed)
        self.assertIn("score/m38b.analytic.metrics.per_person.npz", observed)
        self.assertIn(
            "predictions/folds/m38b.tcn.POS_d2.fold2.seed3301.checkpoint.pt",
            observed,
        )

    def test_prelaunch_contract_must_be_byte_identical_to_published_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / subject.EXPECTED_GIT_COMMIT[:8]
            # The run-directory name is itself part of the prelaunch binding.
            root = root.parent / "m38b-r0-oof-models-20260903c"
            contract = {
                "source_manifest_sha256": "1" * 64,
                "scope": {"chromosome": "22", "root": "R0", "partition": "FIT",
                          "people": 96, "valid_opened": False, "test_opened": False},
            }
            published = root / "contract/m38b.model_contract.receipt.json"
            prelaunch = root / "prelaunch/m38b.model_contract.receipt.json"
            write_json(published, contract)
            write_json(prelaunch, contract)
            write_json(root / "prelaunch/run_provenance.receipt.json", {
                "schema_version": "1.0.0",
                "stage": "M38B_PRELAUNCH_SOURCE_BINDING",
                "status": "PASS_CLEAN_COMMIT_BOUND_TO_SOURCE_MANIFEST",
                "run_id": root.name,
                "created_at_utc": "2026-09-04T00:28:47Z",
                "git_branch": "hpc", "git_commit": subject.EXPECTED_GIT_COMMIT,
                "git_worktree_clean": True,
                "origin_push_status": "PENDING_ENVIRONMENT_SECURITY_APPROVAL",
                "model_contract_receipt_sha256": subject.sha256_file(prelaunch),
                "source_manifest_sha256": contract["source_manifest_sha256"],
                "source_manifest_entries": 28,
                "base_contract_sha256": subject.EXPECTED_BASE_CONTRACT_SHA256,
                "amendment_1_sha256": subject.EXPECTED_AMENDMENT_SHA256,
                "amendment_2_sha256": subject.EXPECTED_AMENDMENT_2_SHA256,
                "scope": {"chromosome": "22", "root": "R0", "partition": "FIT",
                          "valid_opened": False, "test_opened": False},
                "note": "fixture",
            })
            provenance_path = root / "prelaunch/run_provenance.receipt.json"
            expected_provenance_hash = subject.sha256_file(provenance_path)
            with patch.object(subject, "EXPECTED_RUN_PROVENANCE_SHA256",
                              expected_provenance_hash):
                subject.verify_prelaunch(root, contract, subject.HashLedger(root))
                prelaunch.write_text(prelaunch.read_text(encoding="utf-8") + "\n",
                                     encoding="utf-8")
                with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                            "not byte-identical"):
                    subject.verify_prelaunch(root, contract, subject.HashLedger(root))

    def test_prelaunch_provenance_hash_is_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "m38b-r0-oof-models-20260903c"
            contract = {"source_manifest_sha256": "1" * 64}
            published = root / "contract/m38b.model_contract.receipt.json"
            prelaunch = root / "prelaunch/m38b.model_contract.receipt.json"
            write_json(published, contract)
            write_json(prelaunch, contract)
            write_json(root / "prelaunch/run_provenance.receipt.json", {})
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "run-provenance receipt hash differs"):
                subject.verify_prelaunch(root, contract, subject.HashLedger(root))

    def test_bootstrap_contract_rejects_changed_count_or_seed(self) -> None:
        valid = {"bootstrap": {
            "replicates": 10_000, "seed": 38_200_103,
            "unit": "whole person", "stratified_by": "outer fold",
        }}
        subject.validate_fixed_bootstrap(valid, None, "test")
        for field, value, message in (
            ("replicates", 9_999, "test bootstrap replicates must equal 10000"),
            ("seed", 1, "test bootstrap seed must equal 38200103"),
        ):
            altered = deepcopy(valid)
            altered["bootstrap"][field] = value
            with self.assertRaises(subject.M38BPostVerificationError) as raised:
                subject.validate_fixed_bootstrap(altered, None, "test")
            self.assertEqual(str(raised.exception), message)

    def test_family_gates_are_recomputed_from_numeric_fields(self) -> None:
        report = family_report()
        derived = subject.derive_family_gates(report)
        self.assertTrue(all(value is True for value in derived.values()
                            if isinstance(value, bool)))
        subject.validate_reported_family_gates(report, derived, "analytic")
        report["candidate_incremental_gate"]["pass"] = False
        with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                    "disagrees with numeric inputs"):
            subject.validate_reported_family_gates(report, derived, "analytic")

    def test_zero_primary_boundary_does_not_pass_candidate_gate(self) -> None:
        report = family_report()
        for name in ("RE-RD", "RE-SHAM"):
            report["contrasts"][name] = contrast_row(value=0.0, upper=0.0)
        derived = subject.derive_family_gates(report)
        self.assertFalse(derived["candidate_incremental_gate"])
        self.assertTrue(derived["weighted_uniform_no_sign_reversal"])

    def test_string_false_cannot_be_promoted_to_true(self) -> None:
        report = family_report()
        derived = subject.derive_family_gates(report)
        report["candidate_incremental_gate"]["pass"] = "false"
        with self.assertRaises(subject.M38BPostVerificationError) as raised:
            subject.validate_reported_family_gates(report, derived, "tcn")
        self.assertEqual(
            str(raised.exception),
            "tcn.candidate_incremental_gate must be a JSON boolean",
        )

    def test_positive_grid_requires_unique_ordered_ids(self) -> None:
        report = positive_report()
        self.assertTrue(subject.derive_positive_capacity(report))
        report["logical_ids"][-1] = "POS_d1"
        with self.assertRaises(subject.M38BPostVerificationError) as raised:
            subject.derive_positive_capacity(report)
        self.assertEqual(
            str(raised.exception),
            "positive-control IDs must be exact, ordered, and unique",
        )

    def test_positive_capacity_boolean_is_not_trusted(self) -> None:
        report = positive_report()
        report["capacity_gate"]["pass"] = False
        with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                    "disagrees with numeric contrasts"):
            subject.derive_positive_capacity(report)
        report["capacity_gate"]["pass"] = "false"
        with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                    "must be a JSON boolean"):
            subject.derive_positive_capacity(report)

    def test_capacity_control_does_not_replace_real_candidate_gate(self) -> None:
        passed = subject.derive_family_gates(family_report())
        failed = dict(passed)
        failed["candidate_incremental_gate"] = False
        decision = subject.derive_final_document(
            {"analytic": passed, "tcn": failed}, True,
            {name: "a" * 64 for name in subject.PROVENANCE_FIELDS},
        )
        self.assertTrue(decision["families"]["analytic"]["incremental_information_supported"])
        self.assertFalse(decision["families"]["tcn"]["incremental_information_supported"])
        self.assertEqual(decision["families"]["tcn"]["status"], "NOT_SUPPORTED")
        decision = subject.derive_final_document(
            {"analytic": passed, "tcn": passed}, False,
            {name: "a" * 64 for name in subject.PROVENANCE_FIELDS},
        )
        self.assertEqual(decision["families"]["tcn"]["status"], "CAPACITY_INCONCLUSIVE")

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.json"
            path.write_text('{"gate": true, "gate": false}', encoding="utf-8")
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "duplicate JSON key"):
                subject.strict_json_load(path)
            path.write_text('{"metric": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "non-finite JSON constant"):
                subject.strict_json_load(path)

    def test_hash_ledger_detects_a_file_that_changes_during_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "score/result.json"
            write_json(path, {"value": 1})
            ledger = subject.HashLedger(root)
            ledger.audit(path)
            write_json(path, {"value": 2})
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "changed while auditing"):
                ledger.audit(path)

    def test_axis_comparison_is_exact_including_physical_position(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first, second = root / "first.npz", root / "second.npz"
            payload = {
                "sample_key_sha256": np.asarray(["p0", "p1"]),
                "fold_ids": np.asarray([0, 1], dtype=np.uint8),
                "marker_pos": np.asarray([10, 20], dtype=np.int64),
                "marker_cM": np.asarray([0.0, 0.1], dtype=np.float64),
                "state_names": np.asarray(subject.STATE_NAMES),
            }
            write_npz(first, payload)
            write_npz(second, payload)
            subject.validate_axis_identity([first, second], "grid")
            altered = dict(payload)
            altered["marker_pos"] = np.asarray([10, 21], dtype=np.int64)
            second.unlink()
            write_npz(second, altered)
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "axes differ"):
                subject.validate_axis_identity([first, second], "grid")

    def test_positive_materializations_are_bound_directly_to_real_re(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_positive_materializations(root)
            ledger = subject.HashLedger(root)
            event_hash, mask_hash = subject._verify_positive_materializations(root, ledger)
            self.assertEqual(len(event_hash), 64)
            self.assertEqual(len(mask_hash), 64)

    def test_rebound_positive_event_drift_still_fails_against_real_re(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_positive_materializations(root)
            path = root / "controls/positive/features/m38b.POS_d1.fold1.npz"
            receipt_path = root / "controls/positive/features/m38b.POS_d1.fold1.receipt.json"
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            payload["event_locus"] = payload["event_locus"].copy()
            payload["event_locus"][0] += 1
            path.unlink()
            write_npz(path, payload)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["output_sha256"] = subject.sha256_file(path)
            receipt["real_event_identity_sha256"] = subject._array_bundle_sha256(
                payload, subject.EVENT_IDENTITY_MEMBERS,
            )
            receipt_path.unlink()
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "not bound to real RE geometry"):
                subject._verify_positive_materializations(root, subject.HashLedger(root))

    def test_positive_control_rejects_a_real_biological_channel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_positive_materializations(root)
            path = root / "controls/positive/features/m38b.POS_d1.fold1.npz"
            receipt_path = root / "controls/positive/features/m38b.POS_d1.fold1.receipt.json"
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            payload["event_genotype"] = payload["event_genotype"].copy()
            payload["event_genotype"][0] = 1
            path.unlink()
            write_npz(path, payload)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["output_sha256"] = subject.sha256_file(path)
            receipt_path.unlink()
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "retains a real biological channel"):
                subject._verify_positive_materializations(root, subject.HashLedger(root))

    def test_oof_is_rebuilt_from_exact_fold_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            provenance = make_analytic_oof(root)
            subject.verify_oof_derivation(
                root, "analytic", "RE", provenance, subject.HashLedger(root),
            )

    def test_prediction_receipt_binds_exact_fold_partitions_for_all_arm_types(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index, identity in enumerate(("RE", "SHAM", "POS_d1")):
                with self.subTest(identity=identity):
                    receipt = make_partition_binding(root, identity, index, 7000 + index)
                    subject.verify_prediction_partition_binding(
                        root, identity, index, receipt, subject.HashLedger(root),
                    )

    def test_prediction_partition_binding_rejects_mutated_hashes_counts_and_truth_access(self) -> None:
        mutations = {
            "fit_features_sha256": "fit_features_sha256 binding differs",
            "score_features_sha256": "score_features_sha256 binding differs",
            "feature_receipt_sha256": "feature_receipt_sha256 binding differs",
            "fit_truth_sha256": "fit_truth_sha256 binding differs",
            "truth_receipt_sha256": "truth_receipt_sha256 binding differs",
            "inner_split_seed": "inner split seed differs",
            "train_people": "prediction.train_people must equal 48",
            "select_people": "prediction.select_people must equal 16",
            "score_people": "prediction.score_people must equal 32",
            "score_truth_input": "SCORE truth must remain inaccessible",
        }
        for field, message in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                receipt = make_partition_binding(root, "RE", 0, 7000)
                if field.endswith("_sha256"):
                    receipt[field] = "0" * 64
                elif field == "score_truth_input":
                    receipt[field] = "forbidden.score.truth.npz"
                else:
                    receipt[field] += 1
                with self.assertRaisesRegex(subject.M38BPostVerificationError, message):
                    subject.verify_prediction_partition_binding(
                        root, "RE", 0, receipt, subject.HashLedger(root),
                    )

    def test_rebound_oof_drift_fails_exact_fold_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            provenance = make_analytic_oof(root)
            oof = root / "predictions/oof/m38b.analytic.RE.oof.npz"
            receipt_path = root / "predictions/oof/m38b.analytic.RE.oof.receipt.json"
            with np.load(oof, allow_pickle=False) as archive:
                payload = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            payload["probabilities"] = payload["probabilities"].copy()
            payload["probabilities"][0, 0, 0] += np.float32(0.01)
            oof.unlink()
            write_npz(oof, payload)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["output_sha256"] = subject.sha256_file(oof)
            receipt_path.unlink()
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "not the exact fold/seed aggregation"):
                subject.verify_oof_derivation(
                    root, "analytic", "RE", provenance, subject.HashLedger(root),
                )

    def test_npz_recalculation_comparison_detects_bootstrap_index_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "metrics.npz"
            expected = {
                "bootstrap_person_indices": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
                "bootstrap_primary_deltas": np.asarray([[0.1], [0.2]], dtype=np.float64),
            }
            write_npz(path, expected)
            subject._same_npz(expected, path, "score")
            altered = {name: value.copy() for name, value in expected.items()}
            altered["bootstrap_person_indices"][0, 0] = 1
            path.unlink()
            write_npz(path, altered)
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "bootstrap_person_indices values differ"):
                subject._same_npz(expected, path, "score")

    def test_npz_comparison_supports_mixed_exact_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "mixed.npz"
            expected = {
                "labels": np.asarray(["AFR", "EUR", "NAM"]),
                "sample_keys": np.asarray([b"a", b"b"], dtype="S1"),
                "indices": np.asarray([0, 2], dtype=np.int64),
                "mask": np.asarray([True, False], dtype=np.bool_),
                "scores": np.asarray([0.5, np.nan], dtype=np.float64),
            }
            write_npz(path, expected)
            subject._same_npz(expected, path, "mixed")

            altered = {name: value.copy() for name, value in expected.items()}
            altered["labels"][2] = "EUR"
            path.unlink()
            write_npz(path, altered)
            with self.assertRaisesRegex(subject.M38BPostVerificationError,
                                        "mixed.labels values differ"):
                subject._same_npz(expected, path, "mixed")


if __name__ == "__main__":
    unittest.main()
